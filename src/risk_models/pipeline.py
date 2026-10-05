"""
Master Risk Intelligence Pipeline for PPOI (Phase 8).
Orchestrates end-to-end training, validation, model selection, calibration,
out-of-time test evaluation, exposure integration, database population, and reporting.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.risk_models.explainability import explain_operational_risk
from src.risk_models.exposure import build_operational_exposure_layer
from src.risk_models.inventory_risk import (
    InventoryRiskClassifier,
    NaiveInventoryRiskBaseline,
    compute_inventory_risk_score,
)
from src.risk_models.metrics import evaluate_risk_classification
from src.risk_models.supplier_risk import (
    HistoricalSupplierBaseline,
    SupplierRiskClassifier,
    compute_supplier_risk_score,
)
from src.risk_models.validation import (
    FORBIDDEN_INVENTORY_COLUMNS,
    FORBIDDEN_SUPPLIER_COLUMNS,
    INVENTORY_FEATURE_WHITELIST,
    SUPPLIER_FEATURE_WHITELIST,
    assert_risk_feature_whitelist,
    verify_future_perturbation_invariance,
)
from src.utils.logger import get_logger

logger = get_logger("RiskPipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def populate_risk_database_tables(
    inv_preds_df: pd.DataFrame,
    sup_preds_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Populates materialized analytics risk tables in SQLite with proper indexes.
    Preserves star schema integrity.
    """
    if db_path is None:
        db_path = PROJECT_ROOT / "database" / "operations.db"
        
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    logger.info("Materializing analytics risk tables in %s...", db_path)

    # 1. analytics_inventory_risk
    cur.execute("DROP TABLE IF EXISTS analytics_inventory_risk;")
    cur.execute("""
    CREATE TABLE analytics_inventory_risk (
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        warehouse_id TEXT NOT NULL,
        category TEXT,
        stockout_probability_7d REAL,
        inventory_risk_score REAL,
        inventory_risk_band TEXT,
        forecast_demand_7d REAL,
        inventory_position REAL,
        inventory_days_of_supply REAL,
        safety_stock_target REAL,
        safety_stock_gap REAL,
        actual_stockout_within_7d INTEGER,
        split TEXT,
        created_at TEXT,
        PRIMARY KEY (date, product_id, warehouse_id)
    );
    """)
    cur.execute("CREATE INDEX idx_inv_risk_date ON analytics_inventory_risk(date);")
    cur.execute("CREATE INDEX idx_inv_risk_band ON analytics_inventory_risk(inventory_risk_band);")
    cur.execute("CREATE INDEX idx_inv_risk_sku_wh ON analytics_inventory_risk(product_id, warehouse_id);")

    # 2. analytics_supplier_risk
    cur.execute("DROP TABLE IF EXISTS analytics_supplier_risk;")
    cur.execute("""
    CREATE TABLE analytics_supplier_risk (
        po_id TEXT PRIMARY KEY,
        order_date TEXT NOT NULL,
        supplier_id TEXT NOT NULL,
        supplier_name TEXT,
        warehouse_id TEXT,
        product_id TEXT,
        quantity_ordered REAL,
        supplier_delay_probability REAL,
        supplier_risk_score REAL,
        supplier_risk_band TEXT,
        hist_on_time_rate REAL,
        recent_delay_30d REAL,
        supplier_capacity_pressure REAL,
        actual_delay_days INTEGER,
        actual_supplier_late INTEGER,
        split TEXT,
        created_at TEXT
    );
    """)
    cur.execute("CREATE INDEX idx_sup_risk_date ON analytics_supplier_risk(order_date);")
    cur.execute("CREATE INDEX idx_sup_risk_supp ON analytics_supplier_risk(supplier_id);")
    cur.execute("CREATE INDEX idx_sup_risk_band ON analytics_supplier_risk(supplier_risk_band);")

    # 3. analytics_operational_exposure
    cur.execute("DROP TABLE IF EXISTS analytics_operational_exposure;")
    cur.execute("""
    CREATE TABLE analytics_operational_exposure (
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        product_name TEXT,
        category TEXT,
        criticality TEXT,
        warehouse_id TEXT NOT NULL,
        warehouse_name TEXT,
        primary_supplier_id TEXT,
        primary_supplier_name TEXT,
        forecast_demand_7d REAL,
        inventory_position REAL,
        inventory_days_of_supply REAL,
        stockout_probability_7d REAL,
        inventory_risk_score REAL,
        supplier_delay_probability REAL,
        supplier_risk_score REAL,
        operational_exposure_score REAL,
        operational_exposure_band TEXT,
        projected_exposure_quantity REAL,
        projected_exposure_cost REAL,
        explanation_summary TEXT,
        recommended_action TEXT,
        evidence_bullets TEXT,
        created_at TEXT,
        PRIMARY KEY (date, product_id, warehouse_id)
    );
    """)
    cur.execute("CREATE INDEX idx_exp_date ON analytics_operational_exposure(date);")
    cur.execute("CREATE INDEX idx_exp_band ON analytics_operational_exposure(operational_exposure_band);")
    cur.execute("CREATE INDEX idx_exp_sku_wh ON analytics_operational_exposure(product_id, warehouse_id);")
    cur.execute("CREATE INDEX idx_exp_supp ON analytics_operational_exposure(primary_supplier_id);")

    now_str = datetime.now().isoformat()

    # Insert inventory records
    inv_cols = [
        "date", "product_id", "warehouse_id", "category",
        "stockout_probability_7d", "inventory_risk_score", "inventory_risk_band",
        "forecast_demand_7d", "ending_inventory_lag1", "inventory_days_of_supply",
        "safety_stock_target", "safety_stock_gap", "target_stockout_within_7d", "split"
    ]
    inv_insert = inv_preds_df[[c for c in inv_cols if c in inv_preds_df.columns]].copy()
    inv_insert.rename(columns={
        "ending_inventory_lag1": "inventory_position",
        "target_stockout_within_7d": "actual_stockout_within_7d"
    }, inplace=True)
    inv_insert["created_at"] = now_str
    inv_insert.to_sql("analytics_inventory_risk", conn, if_exists="append", index=False)

    # Insert supplier records
    sup_cols = [
        "po_id", "order_date", "supplier_id", "supplier_name", "warehouse_id", "product_id",
        "quantity_ordered", "supplier_delay_probability", "supplier_risk_score", "supplier_risk_band",
        "hist_on_time_rate", "recent_delay_30d", "supplier_capacity_pressure",
        "target_delay_days", "target_supplier_late", "split"
    ]
    sup_insert = sup_preds_df[[c for c in sup_cols if c in sup_preds_df.columns]].copy()
    sup_insert.rename(columns={
        "target_delay_days": "actual_delay_days",
        "target_supplier_late": "actual_supplier_late"
    }, inplace=True)
    sup_insert["created_at"] = now_str
    sup_insert["order_date"] = pd.to_datetime(sup_insert["order_date"]).dt.strftime("%Y-%m-%d")
    sup_insert.to_sql("analytics_supplier_risk", conn, if_exists="append", index=False)

    # Insert operational exposure records
    exp_cols = [
        "date", "product_id", "product_name", "category", "criticality",
        "warehouse_id", "warehouse_name", "primary_supplier_id", "primary_supplier_name",
        "forecast_demand_7d", "ending_inventory_lag1", "inventory_days_of_supply",
        "stockout_probability_7d", "inventory_risk_score",
        "supplier_delay_probability", "supplier_risk_score",
        "operational_exposure_score", "operational_exposure_band",
        "projected_exposure_quantity", "projected_exposure_cost",
        "explanation_summary", "recommended_action", "evidence_bullets"
    ]
    exp_insert = exposure_df[[c for c in exp_cols if c in exposure_df.columns]].copy()
    exp_insert.rename(columns={"ending_inventory_lag1": "inventory_position"}, inplace=True)
    exp_insert["created_at"] = now_str
    exp_insert.to_sql("analytics_operational_exposure", conn, if_exists="append", index=False)

    conn.commit()
    counts = {
        "analytics_inventory_risk": len(inv_insert),
        "analytics_supplier_risk": len(sup_insert),
        "analytics_operational_exposure": len(exp_insert),
    }
    conn.close()
    logger.info("Materialized analytics tables successfully: %s", counts)
    return counts


def run_risk_pipeline(seed: int = 42) -> Dict[str, Any]:
    """
    Executes the master Phase 8 risk modeling pipeline.
    """
    t_start = time.time()
    logger.info("================================================================================")
    logger.info("STARTING PHASE 8: INVENTORY & SUPPLIER RISK MODELING PIPELINE")
    logger.info("================================================================================")
    
    # -------------------------------------------------------------------------
    # 1. Load Certified Features & Incorporate Locked Phase 7 Forecasts
    # -------------------------------------------------------------------------
    t_data0 = time.time()
    logger.info("Loading certified Phase 6 feature stores and locked Phase 7 forecast models...")
    inv_file = PROJECT_ROOT / "data" / "processed" / "features" / "inventory_risk_features.parquet"
    sup_file = PROJECT_ROOT / "data" / "processed" / "features" / "supplier_risk_features.parquet"
    dem_file = PROJECT_ROOT / "data" / "processed" / "features" / "demand_features.parquet"
    fc_model_file = PROJECT_ROOT / "models" / "forecasting" / "model_t+7_xgboost.joblib"

    if not inv_file.exists() or not sup_file.exists() or not fc_model_file.exists():
        raise FileNotFoundError("Prerequisite Phase 6/7 feature stores or models missing.")

    inv_df = pd.read_parquet(inv_file)
    sup_df = pd.read_parquet(sup_file)
    dem_df = pd.read_parquet(dem_file)
    fc_bundle = joblib.load(fc_model_file)

    # Attach locked 7-day demand forecast to inventory feature frame
    logger.info("Generating locked Phase 7 7-day demand forecasts...")
    X_dem = dem_df[fc_bundle["numeric_features"] + fc_bundle["categorical_features"]]
    X_dem_trans = fc_bundle["preprocessor"].transform(X_dem)
    inv_df["forecast_demand_7d"] = np.maximum(0.0, np.round(fc_bundle["model"].predict(X_dem_trans), 2))
    t_data_load = time.time() - t_data0

    # Ensure dates
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    sup_df["order_date"] = pd.to_datetime(sup_df["order_date"])

    # -------------------------------------------------------------------------
    # 2. Chronological Splits & Feature Whitelisting
    # -------------------------------------------------------------------------
    logger.info("Partitioning chronological temporal splits...")
    # Inventory masks
    inv_train_mask = (inv_df["date"] <= "2025-03-31") & (inv_df["has_sufficient_history_28d"] == 1) & (inv_df["is_target_available_7d"] == 1)
    inv_val_mask = (inv_df["date"] >= "2025-04-01") & (inv_df["date"] <= "2025-06-30") & (inv_df["is_target_available_7d"] == 1)
    inv_test_mask = (inv_df["date"] >= "2025-07-01") & (inv_df["date"] <= "2025-12-31") & (inv_df["is_target_available_7d"] == 1)

    inv_features = [c for c in INVENTORY_FEATURE_WHITELIST if c in inv_df.columns]
    assert_risk_feature_whitelist(
        inv_df[inv_features],
        INVENTORY_FEATURE_WHITELIST,
        FORBIDDEN_INVENTORY_COLUMNS,
        context_name="InventoryRisk"
    )

    X_inv_train = inv_df.loc[inv_train_mask, inv_features]
    y_inv_train = inv_df.loc[inv_train_mask, "target_stockout_within_7d"].astype(int)
    X_inv_val = inv_df.loc[inv_val_mask, inv_features]
    y_inv_val = inv_df.loc[inv_val_mask, "target_stockout_within_7d"].astype(int)
    X_inv_test = inv_df.loc[inv_test_mask, inv_features]
    y_inv_test = inv_df.loc[inv_test_mask, "target_stockout_within_7d"].astype(int)

    # Supplier masks & target
    sup_train_mask = (sup_df["order_date"] <= "2025-03-31")
    sup_val_mask = (sup_df["order_date"] >= "2025-04-01") & (sup_df["order_date"] <= "2025-06-30")
    sup_test_mask = (sup_df["order_date"] >= "2025-07-01") & (sup_df["order_date"] <= "2025-12-31")

    sup_features = [c for c in SUPPLIER_FEATURE_WHITELIST if c in sup_df.columns]
    assert_risk_feature_whitelist(
        sup_df[sup_features],
        SUPPLIER_FEATURE_WHITELIST,
        FORBIDDEN_SUPPLIER_COLUMNS,
        context_name="SupplierRisk"
    )

    sup_df["target_supplier_late"] = (sup_df["target_on_time"] == 0).astype(int)
    X_sup_train = sup_df.loc[sup_train_mask, sup_features]
    y_sup_train = sup_df.loc[sup_train_mask, "target_supplier_late"].astype(int)
    X_sup_val = sup_df.loc[sup_val_mask, sup_features]
    y_sup_val = sup_df.loc[sup_val_mask, "target_supplier_late"].astype(int)
    X_sup_test = sup_df.loc[sup_test_mask, sup_features]
    y_sup_test = sup_df.loc[sup_test_mask, "target_supplier_late"].astype(int)

    # -------------------------------------------------------------------------
    # 3. Train & Evaluate Candidate Inventory Risk Models
    # -------------------------------------------------------------------------
    logger.info("Evaluating candidate inventory risk models...")
    inv_candidates = ["naive", "logistic", "random_forest", "xgboost"]
    inv_val_results = []
    inv_trained_models = {}
    timing_inv_train = {}

    for fam in inv_candidates:
        t0 = time.time()
        clf = InventoryRiskClassifier(model_family=fam, random_state=seed)
        clf.fit(X_inv_train, y_inv_train)
        probs_val = clf.predict_proba(X_inv_val)[:, 1]
        t_fit = time.time() - t0
        timing_inv_train[fam] = round(t_fit, 3)

        metrics = evaluate_risk_classification(y_inv_val, probs_val)
        res_row = {
            "domain": "inventory",
            "model": fam,
            "training_time_sec": round(t_fit, 3),
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "val_brier_score": metrics["brier_score"],
            "val_top_10pct_capture": metrics["top_10pct_capture"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_samples": len(y_inv_val),
            "event_rate": metrics["event_rate"],
        }
        inv_val_results.append(res_row)
        inv_trained_models[fam] = clf

    # Select winning inventory model by Validation PR-AUC
    inv_val_df = pd.DataFrame(inv_val_results).sort_values("val_pr_auc", ascending=False)
    selected_inv_model_name = inv_val_df.iloc[0]["model"]
    logger.info("Selected Inventory Model by PR-AUC: %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                selected_inv_model_name, inv_val_df.iloc[0]["val_pr_auc"], inv_val_df.iloc[0]["val_roc_auc"])

    # -------------------------------------------------------------------------
    # 4. Inventory Model Calibration
    # -------------------------------------------------------------------------
    t_cal0 = time.time()
    selected_inv_clf = inv_trained_models[selected_inv_model_name]
    raw_val_brier = inv_val_df.iloc[0]["val_brier_score"]
    
    # Calibrate on validation
    selected_inv_clf.calibrate(X_inv_val, y_inv_val, method="sigmoid")
    cal_inv_probs_val = selected_inv_clf.predict_proba(X_inv_val)[:, 1]
    cal_inv_metrics_val = evaluate_risk_classification(y_inv_val, cal_inv_probs_val)
    cal_val_brier = cal_inv_metrics_val["brier_score"]
    t_inv_cal = time.time() - t_cal0
    logger.info("Inventory calibration: Val Brier reduced from %.4f to %.4f.", raw_val_brier, cal_val_brier)

    # Out-of-Time Test Evaluation
    t_test0 = time.time()
    test_inv_probs = selected_inv_clf.predict_proba(X_inv_test)[:, 1]
    test_inv_metrics = evaluate_risk_classification(y_inv_test, test_inv_probs)
    t_inv_test = time.time() - t_test0
    logger.info("Locked Inventory Model TEST Performance: ROC-AUC=%.4f, PR-AUC=%.4f, Brier=%.4f, Top10=%.4f",
                test_inv_metrics["roc_auc"], test_inv_metrics["pr_auc"], test_inv_metrics["brier_score"], test_inv_metrics["top_10pct_capture"])

    # -------------------------------------------------------------------------
    # 5. Train & Evaluate Candidate Supplier Risk Models
    # -------------------------------------------------------------------------
    logger.info("Evaluating candidate supplier risk models...")
    sup_candidates = ["baseline", "logistic", "random_forest", "xgboost"]
    sup_val_results = []
    sup_trained_models = {}
    timing_sup_train = {}

    for fam in sup_candidates:
        t0 = time.time()
        clf = SupplierRiskClassifier(model_family=fam, random_state=seed)
        clf.fit(X_sup_train, y_sup_train)
        probs_val = clf.predict_proba(X_sup_val)[:, 1]
        t_fit = time.time() - t0
        timing_sup_train[fam] = round(t_fit, 3)

        metrics = evaluate_risk_classification(y_sup_val, probs_val)
        res_row = {
            "domain": "supplier",
            "model": fam,
            "training_time_sec": round(t_fit, 3),
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "val_brier_score": metrics["brier_score"],
            "val_top_10pct_capture": metrics["top_10pct_capture"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_samples": len(y_sup_val),
            "event_rate": metrics["event_rate"],
        }
        sup_val_results.append(res_row)
        sup_trained_models[fam] = clf

    sup_val_df = pd.DataFrame(sup_val_results).sort_values("val_pr_auc", ascending=False)
    selected_sup_model_name = sup_val_df.iloc[0]["model"]
    logger.info("Selected Supplier Model by PR-AUC: %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                selected_sup_model_name, sup_val_df.iloc[0]["val_pr_auc"], sup_val_df.iloc[0]["val_roc_auc"])

    # -------------------------------------------------------------------------
    # 6. Supplier Model Calibration & Test Evaluation
    # -------------------------------------------------------------------------
    t_sup_cal0 = time.time()
    selected_sup_clf = sup_trained_models[selected_sup_model_name]
    raw_sup_val_brier = sup_val_df.iloc[0]["val_brier_score"]
    
    selected_sup_clf.calibrate(X_sup_val, y_sup_val, method="sigmoid")
    cal_sup_probs_val = selected_sup_clf.predict_proba(X_sup_val)[:, 1]
    cal_sup_metrics_val = evaluate_risk_classification(y_sup_val, cal_sup_probs_val)
    cal_sup_val_brier = cal_sup_metrics_val["brier_score"]
    t_sup_cal = time.time() - t_sup_cal0
    logger.info("Supplier calibration: Val Brier reduced from %.4f to %.4f.", raw_sup_val_brier, cal_sup_val_brier)

    t_sup_test0 = time.time()
    test_sup_probs = selected_sup_clf.predict_proba(X_sup_test)[:, 1]
    test_sup_metrics = evaluate_risk_classification(y_sup_test, test_sup_probs)
    t_sup_test = time.time() - t_sup_test0
    logger.info("Locked Supplier Model TEST Performance: ROC-AUC=%.4f, PR-AUC=%.4f, Brier=%.4f, Top10=%.4f",
                test_sup_metrics["roc_auc"], test_sup_metrics["pr_auc"], test_sup_metrics["brier_score"], test_sup_metrics["top_10pct_capture"])

    # -------------------------------------------------------------------------
    # 7. Generate Full Predictions & Scores
    # -------------------------------------------------------------------------
    t_score0 = time.time()
    logger.info("Generating calibrated risk probabilities and operational scores across dataset...")
    # Full inventory predictions (focus on test split for downstream operations & reporting)
    all_inv_test_features = inv_df.loc[inv_test_mask, inv_features]
    inv_test_preds_df = inv_df.loc[inv_test_mask].copy()
    inv_test_preds_df["stockout_probability_7d"] = np.round(selected_inv_clf.predict_proba(all_inv_test_features)[:, 1], 4)
    inv_scores, inv_bands = compute_inventory_risk_score(inv_test_preds_df, inv_test_preds_df["stockout_probability_7d"].values)
    inv_test_preds_df["inventory_risk_score"] = inv_scores
    inv_test_preds_df["inventory_risk_band"] = inv_bands
    inv_test_preds_df["split"] = "TEST"

    # Full supplier predictions for test POs
    all_sup_test_features = sup_df.loc[sup_test_mask, sup_features]
    sup_test_preds_df = sup_df.loc[sup_test_mask].copy()
    sup_test_preds_df["supplier_delay_probability"] = np.round(selected_sup_clf.predict_proba(all_sup_test_features)[:, 1], 4)
    sup_scores, sup_bands = compute_supplier_risk_score(sup_test_preds_df, sup_test_preds_df["supplier_delay_probability"].values)
    sup_test_preds_df["supplier_risk_score"] = sup_scores
    sup_test_preds_df["supplier_risk_band"] = sup_bands
    sup_test_preds_df["split"] = "TEST"

    # Also build latest cross-sectional supplier profile per supplier
    latest_sup_profile = (
        sup_test_preds_df.sort_values("order_date")
        .groupby("supplier_id")
        .last()
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # 8. Integrated Operational Exposure & Prioritization
    # -------------------------------------------------------------------------
    logger.info("Synthesizing integrated operational exposure...")
    conn = get_db_connection()
    dim_product = pd.read_sql_query("SELECT * FROM dim_product", conn)
    dim_supplier = pd.read_sql_query("SELECT * FROM dim_supplier", conn)
    conn.close()

    exposure_df = build_operational_exposure_layer(
        inventory_risk_df=inv_test_preds_df,
        supplier_risk_profile_df=latest_sup_profile,
        dim_product_df=dim_product,
        dim_supplier_df=dim_supplier,
        stockout_penalty_multiplier=1.5,
    )
    t_scoring = time.time() - t_score0

    # -------------------------------------------------------------------------
    # 9. Extract Feature Importances
    # -------------------------------------------------------------------------
    logger.info("Extracting global feature importances...")
    inv_imp_df = selected_inv_clf.get_feature_importances(top_n=15)
    sup_imp_df = selected_sup_clf.get_feature_importances(top_n=15)

    # -------------------------------------------------------------------------
    # 10. Save Artifacts to Disk
    # -------------------------------------------------------------------------
    t_art0 = time.time()
    logger.info("Serializing models, predictions, and reports to disk...")
    risk_models_dir = PROJECT_ROOT / "models" / "risk"
    risk_data_dir = PROJECT_ROOT / "data" / "processed" / "risk"
    reports_dir = PROJECT_ROOT / "reports"
    risk_models_dir.mkdir(parents=True, exist_ok=True)
    risk_data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Models
    joblib.dump(selected_inv_clf, risk_models_dir / "inventory_risk_model.joblib")
    joblib.dump(selected_sup_clf, risk_models_dir / "supplier_risk_model.joblib")

    # Datasets
    # Format date strings for clean export
    inv_test_preds_df["date"] = pd.to_datetime(inv_test_preds_df["date"]).dt.strftime("%Y-%m-%d")
    exposure_df["date"] = pd.to_datetime(exposure_df["date"]).dt.strftime("%Y-%m-%d")
    sup_test_preds_df["order_date"] = pd.to_datetime(sup_test_preds_df["order_date"]).dt.strftime("%Y-%m-%d")

    inv_test_preds_df.to_parquet(risk_data_dir / "inventory_risk_predictions.parquet", index=False)
    inv_test_preds_df.to_csv(risk_data_dir / "inventory_risk_predictions.csv", index=False)

    sup_test_preds_df.to_parquet(risk_data_dir / "supplier_risk_predictions.parquet", index=False)
    sup_test_preds_df.to_csv(risk_data_dir / "supplier_risk_predictions.csv", index=False)

    exposure_df.to_parquet(risk_data_dir / "operational_risk_priorities.parquet", index=False)
    exposure_df.to_csv(risk_data_dir / "operational_risk_priorities.csv", index=False)

    # -------------------------------------------------------------------------
    # 11. Database Population
    # -------------------------------------------------------------------------
    db_counts = populate_risk_database_tables(
        inv_preds_df=inv_test_preds_df,
        sup_preds_df=sup_test_preds_df,
        exposure_df=exposure_df,
    )

    # -------------------------------------------------------------------------
    # 12. Model Comparison Table & Results Manifest
    # -------------------------------------------------------------------------
    comp_df = pd.concat([inv_val_df, sup_val_df], ignore_index=True)
    comp_df.to_csv(reports_dir / "risk_model_comparison.csv", index=False)

    results_manifest = {
        "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "timing_breakdown_sec": {
            "data_loading": round(t_data_load, 3),
            "inventory_training": timing_inv_train,
            "inventory_calibration": round(t_inv_cal, 3),
            "inventory_test": round(t_inv_test, 3),
            "supplier_training": timing_sup_train,
            "supplier_calibration": round(t_sup_cal, 3),
            "supplier_test": round(t_sup_test, 3),
            "risk_scoring_and_exposure": round(t_scoring, 3),
            "total_pipeline_sec": round(time.time() - t_start, 2),
        },
        "inventory_risk_model": {
            "selected_model": selected_inv_model_name,
            "selection_criterion": "Validation PR-AUC",
            "val_roc_auc": inv_val_df.iloc[0]["val_roc_auc"],
            "val_pr_auc": inv_val_df.iloc[0]["val_pr_auc"],
            "val_brier_raw": raw_val_brier,
            "val_brier_calibrated": cal_val_brier,
            "test_metrics": test_inv_metrics,
            "top_features": inv_imp_df.to_dict(orient="records"),
        },
        "supplier_risk_model": {
            "selected_model": selected_sup_model_name,
            "selection_criterion": "Validation PR-AUC",
            "val_roc_auc": sup_val_df.iloc[0]["val_roc_auc"],
            "val_pr_auc": sup_val_df.iloc[0]["val_pr_auc"],
            "val_brier_raw": raw_sup_val_brier,
            "val_brier_calibrated": cal_sup_val_brier,
            "test_metrics": test_sup_metrics,
            "top_features": sup_imp_df.to_dict(orient="records"),
        },
        "operational_exposure_summary": {
            "total_evaluated_records": len(exposure_df),
            "risk_band_counts": exposure_df["operational_exposure_band"].value_counts().to_dict(),
            "critical_risk_count": int((exposure_df["operational_exposure_band"] == "CRITICAL").sum()),
            "high_risk_count": int((exposure_df["operational_exposure_band"] == "HIGH").sum()),
            "total_projected_exposure_cost": float(np.round(exposure_df["projected_exposure_cost"].sum(), 2)),
        },
        "database_materialization": db_counts,
    }

    with open(reports_dir / "risk_model_results.json", "w") as f:
        json.dump(results_manifest, f, indent=2)

    t_total = time.time() - t_start
    logger.info("PHASE 8 MASTER PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS.", t_total)
    return results_manifest
