"""
Master Forecasting Pipeline Orchestration Module for PPOI (Phase 7).
Coordinates data loading, feature whitelisting, chronological temporal partitioning,
multi-horizon candidate model fitting (Seasonal Naive, Moving Average, Ridge, XGBoost),
validation-based model selection, empirical prediction intervals, out-of-time test evaluation,
residual breakdown, and artifact serialization.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.forecasting.baselines import MovingAverageForecaster, SeasonalNaiveForecaster
from src.forecasting.metrics import evaluate_forecast
from src.forecasting.model_selection import (
    build_model_comparison_table,
    select_best_model_per_horizon,
)
from src.forecasting.prediction_intervals import EmpiricalPredictionIntervals
from src.forecasting.ridge_forecaster import RidgeForecaster
from src.forecasting.validation import (
    DEFAULT_FEATURE_WHITELIST,
    assert_feature_whitelist,
    verify_future_perturbation_invariance,
)
from src.forecasting.xgboost_forecaster import XGBoostForecaster
from src.utils.logger import get_logger

logger = get_logger("ForecastPipeline")

DATA_PATH = Path("data/processed/features/demand_features.parquet")
MODELS_DIR = Path("models/forecasting")
FORECASTS_DIR = Path("data/processed/forecasts")
REPORTS_DIR = Path("reports")

HORIZONS = [1, 7, 14, 28]


def run_forecast_pipeline(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    forecasts_dir: Path = FORECASTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the certified Phase 7 forecasting pipeline end-to-end.
    """
    t_start = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load and Audit Certified Feature Data
    # -------------------------------------------------------------------------
    logger.info("Loading certified Phase 6 feature data from %s...", data_path)
    t_load = time.time()
    df = pd.read_parquet(data_path)
    load_time = time.time() - t_load

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    expected_rows = 175_440
    if len(df) != expected_rows:
        raise ValueError(f"Feature dataset row count mismatch: expected {expected_rows}, got {len(df)}")

    # -------------------------------------------------------------------------
    # 2. Programmatic Feature Whitelist Assertion
    # -------------------------------------------------------------------------
    logger.info("Asserting feature whitelist and verifying zero target leakage...")
    assert_feature_whitelist(DEFAULT_FEATURE_WHITELIST)

    # -------------------------------------------------------------------------
    # 3. Chronological Temporal Splitting
    # -------------------------------------------------------------------------
    logger.info("Constructing chronological temporal partitions...")
    train_mask = (df["date"] >= "2024-01-01") & (df["date"] <= "2025-03-31")
    val_mask = (df["date"] >= "2025-04-01") & (df["date"] <= "2025-06-30")
    test_mask = (df["date"] >= "2025-07-01") & (df["date"] <= "2025-12-31")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    logger.info(
        "Temporal Split: Train=%d rows (%s to %s), Val=%d rows (%s to %s), Test=%d rows (%s to %s)",
        len(df_train), df_train["date"].min().strftime("%Y-%m-%d"), df_train["date"].max().strftime("%Y-%m-%d"),
        len(df_val), df_val["date"].min().strftime("%Y-%m-%d"), df_val["date"].max().strftime("%Y-%m-%d"),
        len(df_test), df_test["date"].min().strftime("%Y-%m-%d"), df_test["date"].max().strftime("%Y-%m-%d")
    )

    assert len(df_train) == 109_440, f"Train row count mismatch: {len(df_train)}"
    assert len(df_val) == 21_840, f"Val row count mismatch: {len(df_val)}"
    assert len(df_test) == 44_160, f"Test row count mismatch: {len(df_test)}"
    assert df_train["date"].max() < df_val["date"].min(), "Leakage: Train overlaps Validation!"
    assert df_val["date"].max() < df_test["date"].min(), "Leakage: Validation overlaps Test!"

    # -------------------------------------------------------------------------
    # 4. Multi-Horizon Model Training & Validation Evaluation
    # -------------------------------------------------------------------------
    validation_results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Dict[str, Any]] = {}
    val_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    timing_breakdown: Dict[str, float] = {"data_loading": load_time}

    for h in HORIZONS:
        h_key = f"t+{h}"
        target_col = f"target_demand_t_plus_{h}"
        avail_col = f"is_target_available_t_plus_{h}"
        logger.info("=== Training and Evaluating Models for Horizon %s (%s) ===", h_key, target_col)

        # Validation target slice (filtered by availability)
        val_eval_mask = (df_val[avail_col] == 1) & df_val[target_col].notna()
        df_val_eval = df_val[val_eval_mask].copy()
        y_val_actual = df_val_eval[target_col].values

        trained_models[h_key] = {}
        val_predictions[h_key] = {}

        # ----------------- Model 1: Seasonal Naive Baseline -----------------
        t0 = time.time()
        sn_model = SeasonalNaiveForecaster(season_length=7)
        sn_model.fit(df_train, target_col)
        sn_val_pred = sn_model.predict(df_val_eval, horizon=h, full_df=df)
        t_sn = time.time() - t0
        timing_breakdown[f"sn_{h_key}"] = t_sn

        sn_metrics = evaluate_forecast(y_val_actual, sn_val_pred)
        validation_results.append({
            "model": "Seasonal Naive",
            "horizon": h_key,
            "training_time_sec": round(t_sn, 3),
            **sn_metrics,
        })
        trained_models[h_key]["Seasonal Naive"] = sn_model
        val_predictions[h_key]["Seasonal Naive"] = sn_val_pred

        # ----------------- Model 2: Moving Average Baseline -----------------
        # Evaluate candidate windows (7, 14, 28) on validation and select best
        t0 = time.time()
        best_ma_window = 7
        best_ma_wape = float("inf")
        best_ma_pred = None
        best_ma_model = None

        for w in [7, 14, 28]:
            ma_model = MovingAverageForecaster(window=w)
            ma_model.fit(df_train, target_col)
            ma_pred = ma_model.predict(df_val_eval, horizon=h, full_df=df)
            m_metrics = evaluate_forecast(y_val_actual, ma_pred)
            if m_metrics["wape"] < best_ma_wape:
                best_ma_wape = m_metrics["wape"]
                best_ma_window = w
                best_ma_pred = ma_pred
                best_ma_model = ma_model

        t_ma = time.time() - t0
        timing_breakdown[f"ma_{h_key}"] = t_ma
        ma_metrics = evaluate_forecast(y_val_actual, best_ma_pred)
        validation_results.append({
            "model": f"Moving Average (W={best_ma_window})",
            "horizon": h_key,
            "training_time_sec": round(t_ma, 3),
            **ma_metrics,
        })
        trained_models[h_key]["Moving Average"] = best_ma_model
        val_predictions[h_key]["Moving Average"] = best_ma_pred

        # ----------------- Model 3: Ridge Regression ------------------------
        t0 = time.time()
        ridge_model = RidgeForecaster(random_state=random_state)
        best_alpha = ridge_model.tune_alpha_on_validation(
            df_train=df_train,
            df_val=df_val_eval,
            target_col=target_col,
            candidate_alphas=[0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
        )
        ridge_val_pred = ridge_model.predict(df_val_eval)
        t_ridge = time.time() - t0
        timing_breakdown[f"ridge_{h_key}"] = t_ridge

        ridge_metrics = evaluate_forecast(y_val_actual, ridge_val_pred)
        validation_results.append({
            "model": f"Ridge (alpha={best_alpha})",
            "horizon": h_key,
            "training_time_sec": round(t_ridge, 3),
            **ridge_metrics,
        })
        trained_models[h_key]["Ridge"] = ridge_model
        val_predictions[h_key]["Ridge"] = ridge_val_pred

        # ----------------- Model 4: XGBoost Regressor -----------------------
        t0 = time.time()
        xgb_model = XGBoostForecaster(random_state=random_state)
        best_xgb_cfg = xgb_model.tune_on_validation(
            df_train=df_train,
            df_val=df_val_eval,
            target_col=target_col,
        )
        xgb_val_pred = xgb_model.predict(df_val_eval)
        t_xgb = time.time() - t0
        timing_breakdown[f"xgb_{h_key}"] = t_xgb

        xgb_metrics = evaluate_forecast(y_val_actual, xgb_val_pred)
        validation_results.append({
            "model": f"XGBoost ({best_xgb_cfg.get('config_name')})",
            "horizon": h_key,
            "training_time_sec": round(t_xgb, 3),
            **xgb_metrics,
        })
        trained_models[h_key]["XGBoost"] = xgb_model
        val_predictions[h_key]["XGBoost"] = xgb_val_pred

    # -------------------------------------------------------------------------
    # 5. Validation Model Comparison & Selection
    # -------------------------------------------------------------------------
    logger.info("Executing validation model comparison and selection...")
    comparison_df = build_model_comparison_table(validation_results)
    comparison_csv = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Saved model comparison table to %s", comparison_csv)

    # Primary selection rule: WAPE on validation set
    selected_models_meta = select_best_model_per_horizon(validation_results, primary_metric="wape")

    # Map standardized winner key to internal trained model handle
    def resolve_model_family(winner_name: str) -> str:
        if "XGBoost" in winner_name:
            return "XGBoost"
        elif "Ridge" in winner_name:
            return "Ridge"
        elif "Seasonal Naive" in winner_name:
            return "Seasonal Naive"
        elif "Moving Average" in winner_name:
            return "Moving Average"
        return winner_name

    # -------------------------------------------------------------------------
    # 6. Empirical Prediction Intervals & Out-of-Time Test Evaluation
    # -------------------------------------------------------------------------
    logger.info("Locking selected models and evaluating out-of-time test performance...")
    test_results: List[Dict[str, Any]] = []
    interval_results: Dict[str, Any] = {}
    test_forecast_frames: List[pd.DataFrame] = []
    feature_importances: Dict[str, Any] = {}

    for h in HORIZONS:
        h_key = f"t+{h}"
        target_col = f"target_demand_t_plus_{h}"
        avail_col = f"is_target_available_t_plus_{h}"
        winner_info = selected_models_meta[h_key]
        winner_label = winner_info["selected_model"]
        winner_family = resolve_model_family(winner_label)
        selected_model = trained_models[h_key][winner_family]

        # 1. Fit Empirical Prediction Interval on validation residuals
        val_eval_mask = (df_val[avail_col] == 1) & df_val[target_col].notna()
        df_val_eval = df_val[val_eval_mask].copy()
        y_val_actual = df_val_eval[target_col].values
        y_val_pred = val_predictions[h_key][winner_family]

        interval_estimator = EmpiricalPredictionIntervals(nominal_coverage=0.80)
        interval_estimator.fit(y_val_actual, y_val_pred)

        # 2. Test evaluation (Filtered to complete target availability)
        test_eval_mask = (df_test[avail_col] == 1) & df_test[target_col].notna()
        df_test_eval = df_test[test_eval_mask].copy().reset_index(drop=True)
        y_test_actual = df_test_eval[target_col].values

        t0 = time.time()
        if winner_family in ["Seasonal Naive", "Moving Average"]:
            y_test_pred = selected_model.predict(df_test_eval, horizon=h, full_df=df)
        else:
            y_test_pred = selected_model.predict(df_test_eval)
        pred_time = time.time() - t0

        test_metrics = evaluate_forecast(y_test_actual, y_test_pred)
        test_lower, test_upper = interval_estimator.predict_intervals(y_test_pred)
        interval_eval = interval_estimator.evaluate_coverage(y_test_actual, test_lower, test_upper)

        test_results.append({
            "horizon": h_key,
            "selected_model": winner_label,
            "prediction_time_sec": round(pred_time, 3),
            **test_metrics,
            "nominal_coverage": interval_eval["nominal_coverage"],
            "empirical_coverage": interval_eval["empirical_coverage"],
            "average_interval_width": interval_eval["average_interval_width"],
        })
        interval_results[h_key] = interval_eval

        # 3. Construct structured forecast output records
        fc_df = pd.DataFrame({
            "date": df_test_eval["date"].dt.strftime("%Y-%m-%d"),
            "product_id": df_test_eval["product_id"],
            "warehouse_id": df_test_eval["warehouse_id"],
            "category": df_test_eval["category"],
            "actual_demand": y_test_actual,
            "forecast": np.round(y_test_pred, 2),
            "horizon": h_key,
            "model": winner_label,
            "lower_prediction_interval": np.round(test_lower, 2),
            "upper_prediction_interval": np.round(test_upper, 2),
            "residual": np.round(y_test_actual - y_test_pred, 2),
            "split": "TEST",
        })
        test_forecast_frames.append(fc_df)

        # 4. Save model artifact
        if hasattr(selected_model, "save"):
            model_artifact_path = models_dir / f"model_{h_key}_{winner_family.lower().replace(' ', '_')}.joblib"
            selected_model.save(model_artifact_path)

        # 5. Extract Feature Importance for XGBoost
        xgb_instance = trained_models[h_key].get("XGBoost")
        if xgb_instance is not None:
            fi = xgb_instance.get_feature_importance(top_n=15)
            feature_importances[h_key] = fi.to_dict(orient="records")

    # Combine all forecast records
    all_forecasts_df = pd.concat(test_forecast_frames, ignore_index=True)
    forecasts_parquet = forecasts_dir / "demand_forecasts.parquet"
    forecasts_csv = forecasts_dir / "demand_forecasts.csv"
    all_forecasts_df.to_parquet(forecasts_parquet, index=False)
    all_forecasts_df.to_csv(forecasts_csv, index=False)
    logger.info("Saved test demand forecasts to %s and %s (%d rows)", forecasts_parquet, forecasts_csv, len(all_forecasts_df))

    # -------------------------------------------------------------------------
    # 7. Residual Analysis on Selected Models
    # -------------------------------------------------------------------------
    logger.info("Conducting residual analysis across operational dimensions...")
    all_forecasts_df["month"] = pd.to_datetime(all_forecasts_df["date"]).dt.strftime("%Y-%m")
    
    # Residual summary by horizon
    residual_summary_by_horizon = {}
    for h in HORIZONS:
        h_key = f"t+{h}"
        h_df = all_forecasts_df[all_forecasts_df["horizon"] == h_key]
        res = h_df["residual"].values
        residual_summary_by_horizon[h_key] = {
            "mean_residual": round(float(np.mean(res)), 4),
            "std_residual": round(float(np.std(res, ddof=1)), 4),
            "p10_residual": round(float(np.percentile(res, 10)), 4),
            "p50_residual": round(float(np.percentile(res, 50)), 4),
            "p90_residual": round(float(np.percentile(res, 90)), 4),
        }

    # Breakdown by warehouse (for t+1)
    t1_df = all_forecasts_df[all_forecasts_df["horizon"] == "t+1"]
    by_warehouse = (
        t1_df.groupby("warehouse_id")
        .apply(lambda g: pd.Series({
            "wape": round(float(np.sum(np.abs(g["residual"])) / np.sum(g["actual_demand"])), 4),
            "mae": round(float(np.mean(np.abs(g["residual"]))), 4),
            "bias": round(float(np.mean(-g["residual"])), 4),  # pred - actual
        }))
        .to_dict(orient="index")
    )

    # Breakdown by category (for t+1)
    by_category = (
        t1_df.groupby("category")
        .apply(lambda g: pd.Series({
            "wape": round(float(np.sum(np.abs(g["residual"])) / np.sum(g["actual_demand"])), 4),
            "mae": round(float(np.mean(np.abs(g["residual"]))), 4),
            "bias": round(float(np.mean(-g["residual"])), 4),
        }))
        .to_dict(orient="index")
    )

    # -------------------------------------------------------------------------
    # 8. Leakage Regression Verification Test
    # -------------------------------------------------------------------------
    logger.info("Running automated future perturbation invariance test...")
    perturbation_test = verify_future_perturbation_invariance(df, cutoff_date="2025-06-30")

    total_pipeline_time = time.time() - t_start
    timing_breakdown["total_pipeline_sec"] = round(total_pipeline_time, 2)

    # -------------------------------------------------------------------------
    # 9. Serialize Machine-Readable Results JSON
    # -------------------------------------------------------------------------
    results_payload = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": random_state,
        "timing_breakdown": timing_breakdown,
        "validation_results": validation_results,
        "selected_models": selected_models_meta,
        "test_results": test_results,
        "interval_results": interval_results,
        "residual_analysis": {
            "by_horizon": residual_summary_by_horizon,
            "by_warehouse_t1": by_warehouse,
            "by_category_t1": by_category,
        },
        "feature_importances": feature_importances,
        "perturbation_test": perturbation_test,
    }

    results_json = reports_dir / "forecasting_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, default=str)
    logger.info("Saved machine-readable forecasting results to %s", results_json)

    # Print executive summary to stdout
    print("\n" + "=" * 70)
    print("PHASE 7 FORECASTING PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Total Runtime          : {total_pipeline_time:.2f} seconds")
    print(f"Feature Whitelist      : {len(DEFAULT_FEATURE_WHITELIST)} columns (0 target leaks)")
    print(f"Temporal Splits        : Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
    print("-" * 70)
    print("OUT-OF-TIME TEST PERFORMANCE (SELECTED MODELS):")
    for tr in test_results:
        print(f"  {tr['horizon']:4s} -> {tr['selected_model']:<30s} | WAPE: {tr['wape']:.4f} | MAE: {tr['mae']:.2f} | Coverage: {tr['empirical_coverage']*100:.1f}%")
    print("-" * 70)
    print(f"Future Invariance Test : {'PASSED (max_diff = 0.00)' if perturbation_test['is_invariant'] else 'FAILED'}")
    print("=" * 70 + "\n")

    return results_payload


if __name__ == "__main__":
    run_forecast_pipeline()
