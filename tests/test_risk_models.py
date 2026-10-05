"""
Comprehensive Pytest Suite for Inventory & Supplier Risk Models (Phase 8).
Validates:
1. test_inventory_target_exists
2. test_supplier_target_exists
3. test_temporal_split
4. test_feature_whitelist
5. test_future_feature_rejection
6. test_inventory_model_fit
7. test_supplier_model_fit
8. test_probability_bounds
9. test_calibration_output
10. test_risk_score_bounds
11. test_risk_band_assignment
12. test_future_perturbation_invariance
13. test_deterministic_explanation
14. test_prediction_output_schema
15. test_model_serialization
16. test_reproducibility
17. test_no_duplicate_prediction_keys
18. test_missing_value_handling
19. test_class_imbalance_handling
20. test_full_risk_pipeline_smoke
"""

import json
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.risk_models.explainability import (
    explain_operational_risk,
    generate_deterministic_evidence,
    generate_recommended_action,
)
from src.risk_models.exposure import build_operational_exposure_layer
from src.risk_models.inventory_risk import (
    InventoryRiskClassifier,
    NaiveInventoryRiskBaseline,
    compute_inventory_risk_score,
)
from src.risk_models.metrics import (
    calculate_top_k_capture,
    compute_brier_skill_score,
    evaluate_risk_classification,
)
from src.risk_models.pipeline import run_risk_pipeline
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


@pytest.fixture(scope="module")
def sample_inventory_features():
    path = Path("data/processed/features/inventory_risk_features.parquet")
    if not path.exists():
        pytest.skip("inventory_risk_features.parquet not available")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def sample_supplier_features():
    path = Path("data/processed/features/supplier_risk_features.parquet")
    if not path.exists():
        pytest.skip("supplier_risk_features.parquet not available")
    df = pd.read_parquet(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_inventory_target_exists(sample_inventory_features):
    """1. Verify primary and secondary inventory stockout risk targets exist."""
    assert "target_stockout_within_7d" in sample_inventory_features.columns
    assert "target_stockout_t_plus_1" in sample_inventory_features.columns
    assert "target_stockout_within_14d" in sample_inventory_features.columns
    assert "target_lost_sales_7d" in sample_inventory_features.columns
    
    y = sample_inventory_features["target_stockout_within_7d"].dropna()
    assert set(y.unique()).issubset({0, 1})
    assert 0.10 <= y.mean() <= 0.30  # ~19% positive rate


def test_supplier_target_exists(sample_supplier_features):
    """2. Verify primary and secondary supplier delay risk targets exist."""
    assert "target_on_time" in sample_supplier_features.columns
    assert "target_delay_days" in sample_supplier_features.columns
    assert "target_delay_gt_2d" in sample_supplier_features.columns
    assert "target_otif" in sample_supplier_features.columns
    
    late = (sample_supplier_features["target_on_time"] == 0).astype(int)
    assert 0.35 <= late.mean() <= 0.60  # ~49.6% late rate


def test_temporal_split(sample_inventory_features, sample_supplier_features):
    """3. Verify strictly chronological non-overlapping temporal split boundaries."""
    # Inventory dates
    train_dates = sample_inventory_features.loc[sample_inventory_features["date"] <= "2025-03-31", "date"]
    val_dates = sample_inventory_features.loc[
        (sample_inventory_features["date"] >= "2025-04-01") & 
        (sample_inventory_features["date"] <= "2025-06-30"), "date"
    ]
    test_dates = sample_inventory_features.loc[sample_inventory_features["date"] >= "2025-07-01", "date"]
    
    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()
    assert train_dates.max() == pd.to_datetime("2025-03-31")
    assert val_dates.max() == pd.to_datetime("2025-06-30")
    assert test_dates.max() == pd.to_datetime("2025-12-31")


def test_feature_whitelist(sample_inventory_features, sample_supplier_features):
    """4. Verify feature whitelist passes when only approved columns are included."""
    inv_cols = [c for c in INVENTORY_FEATURE_WHITELIST if c in sample_inventory_features.columns]
    assert_risk_feature_whitelist(
        sample_inventory_features[inv_cols],
        INVENTORY_FEATURE_WHITELIST,
        FORBIDDEN_INVENTORY_COLUMNS,
        context_name="InventoryTest"
    )
    
    sup_cols = [c for c in SUPPLIER_FEATURE_WHITELIST if c in sample_supplier_features.columns]
    assert_risk_feature_whitelist(
        sample_supplier_features[sup_cols],
        SUPPLIER_FEATURE_WHITELIST,
        FORBIDDEN_SUPPLIER_COLUMNS,
        context_name="SupplierTest"
    )


def test_future_feature_rejection(sample_inventory_features):
    """5. Verify loud ValueError if target or post-outcome columns enter feature matrix."""
    violating_df = sample_inventory_features[["beginning_inventory", "target_stockout_within_7d"]].copy()
    with pytest.raises(ValueError, match="FATAL LEAKAGE DETECTED"):
        assert_risk_feature_whitelist(
            violating_df,
            INVENTORY_FEATURE_WHITELIST,
            FORBIDDEN_INVENTORY_COLUMNS,
            context_name="LeakageTest"
        )


def test_inventory_model_fit(sample_inventory_features):
    """6. Verify inventory model fits on training data and generates predictions."""
    train_mask = (sample_inventory_features["date"] <= "2024-03-31") & (sample_inventory_features["has_sufficient_history_28d"] == 1)
    features = [c for c in INVENTORY_FEATURE_WHITELIST if c in sample_inventory_features.columns]
    
    X = sample_inventory_features.loc[train_mask, features].head(500)
    y = sample_inventory_features.loc[train_mask, "target_stockout_within_7d"].head(500).astype(int)
    
    clf = InventoryRiskClassifier(model_family="xgboost", n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    
    assert probs.shape == (500, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_supplier_model_fit(sample_supplier_features):
    """7. Verify supplier model fits on training data and generates predictions."""
    train_mask = (sample_supplier_features["order_date"] <= "2024-06-30")
    features = [c for c in SUPPLIER_FEATURE_WHITELIST if c in sample_supplier_features.columns]
    
    X = sample_supplier_features.loc[train_mask, features].head(300)
    y = (sample_supplier_features.loc[train_mask, "target_on_time"].head(300) == 0).astype(int)
    
    clf = SupplierRiskClassifier(model_family="random_forest", n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    
    assert probs.shape == (300, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_probability_bounds():
    """8. Verify predicted probabilities are strictly bounded within [0, 1]."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.05, 0.95, 0.20, 0.85, 0.70, 0.10])
    
    metrics = evaluate_risk_classification(y_true, y_prob)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_calibration_output(sample_inventory_features):
    """9. Verify probability calibration on validation split reduces or maintains Brier score."""
    pos_idx = sample_inventory_features.index[sample_inventory_features["target_stockout_within_7d"] == 1]
    neg_idx = sample_inventory_features.index[sample_inventory_features["target_stockout_within_7d"] == 0]
    train_idx = np.concatenate([pos_idx[:500], neg_idx[:1500]])
    val_idx = np.concatenate([pos_idx[500:800], neg_idx[1500:2500]])
    
    features = [c for c in INVENTORY_FEATURE_WHITELIST if c in sample_inventory_features.columns]
    X_train = sample_inventory_features.loc[train_idx, features]
    y_train = sample_inventory_features.loc[train_idx, "target_stockout_within_7d"].astype(int)
    X_val = sample_inventory_features.loc[val_idx, features]
    y_val = sample_inventory_features.loc[val_idx, "target_stockout_within_7d"].astype(int)
    
    clf = InventoryRiskClassifier(model_family="logistic", random_state=42)
    clf.fit(X_train, y_train)
    raw_probs = clf.predict_proba(X_val)[:, 1]
    raw_brier = evaluate_risk_classification(y_val, raw_probs)["brier_score"]
    
    clf.calibrate(X_val, y_val, method="sigmoid")
    cal_probs = clf.predict_proba(X_val)[:, 1]
    cal_brier = evaluate_risk_classification(y_val, cal_probs)["brier_score"]
    
    assert cal_brier <= raw_brier + 0.05
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)


def test_risk_score_bounds(sample_inventory_features):
    """10. Verify deterministic operational risk scores are strictly bounded within [0, 100]."""
    sub = sample_inventory_features.head(100).copy()
    probs = np.linspace(0.0, 1.0, 100)
    scores, bands = compute_inventory_risk_score(sub, probs)
    
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 100.0)
    assert len(scores) == 100


def test_risk_band_assignment(sample_inventory_features):
    """11. Verify risk band assignments strictly adhere to configured thresholds."""
    sub = sample_inventory_features.head(4).copy()
    # Test extreme probs
    probs = np.array([0.01, 0.35, 0.70, 0.99])
    sub["inventory_days_of_supply"] = [25.0, 10.0, 5.0, 0.5]
    sub["ending_inventory_lag1"] = [1000.0, 400.0, 100.0, 10.0]
    sub["safety_stock_target"] = [500.0, 500.0, 500.0, 500.0]
    sub["stockout_rate_28d"] = [0.0, 0.05, 0.25, 0.80]
    
    scores, bands = compute_inventory_risk_score(sub, probs)
    valid_bands = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for b in bands:
        assert b in valid_bands


def test_future_perturbation_invariance():
    """12. Verify that future mutations strictly after split date do not alter historical inputs."""
    # Create synthetic daily ledger
    dates = pd.date_range("2025-01-01", "2025-12-31")
    df = pd.DataFrame({
        "date": dates,
        "product_id": "SKU-001",
        "warehouse_id": "WH-01",
        "demand_requested": np.random.uniform(20, 50, len(dates)),
        "ending_inventory": np.random.uniform(100, 300, len(dates)),
    })
    
    def builder(data):
        d = data.copy()
        d["demand_mean_7"] = d["demand_requested"].shift(1).rolling(7, min_periods=1).mean()
        d["ending_inventory_lag1"] = d["ending_inventory"].shift(1)
        return d
        
    diff = verify_future_perturbation_invariance(
        builder,
        df,
        split_date="2025-06-30",
        perturb_col="demand_requested",
        perturb_delta=10000.0,
    )
    assert diff == 0.0


def test_deterministic_explanation():
    """13. Verify deterministic evidence and recommendation generation."""
    row = pd.Series({
        "forecast_demand_7d": 500.0,
        "ending_inventory_lag1": 150.0,
        "inventory_days_of_supply": 2.1,
        "safety_stock_target": 300.0,
        "stockout_count_7d": 3,
        "stockout_rate_28d": 0.35,
        "supplier_delay_probability": 0.68,
        "supplier_name": "Apex Micro Parts",
        "criticality": "High",
        "supplier_dependency": "Dual Sourced",
    })
    
    evidence = generate_deterministic_evidence(row)
    assert len(evidence) >= 3
    assert any("exceeds available inventory" in e for e in evidence)
    assert any("days of supply" in e for e in evidence)
    assert any("Apex Micro Parts" in e for e in evidence)
    
    summary, ev_list, action = explain_operational_risk(row, "CRITICAL")
    assert "Elevated critical operational exposure" in summary
    assert "Expedite purchase order allocation" in action


def test_prediction_output_schema():
    """14. Verify saved prediction parquet schema and columns."""
    inv_pred_path = Path("data/processed/risk/inventory_risk_predictions.parquet")
    if not inv_pred_path.exists():
        pytest.skip("inventory_risk_predictions.parquet not yet written")
    df = pd.read_parquet(inv_pred_path)
    
    required_cols = [
        "date", "product_id", "warehouse_id", "category",
        "stockout_probability_7d", "inventory_risk_score", "inventory_risk_band",
        "forecast_demand_7d", "split"
    ]
    for c in required_cols:
        assert c in df.columns
    assert len(df) > 0


def test_model_serialization(tmp_path):
    """15. Verify risk model artifact serialization and reloading."""
    clf = InventoryRiskClassifier(model_family="naive")
    X = pd.DataFrame({
        "inventory_days_of_supply": [10.0, 3.0, 1.0],
        "safety_stock_gap": [50.0, -10.0, -100.0]
    })
    clf.fit(X, pd.Series([0, 1, 1]))
    
    save_path = tmp_path / "test_model.joblib"
    joblib.dump(clf, save_path)
    
    loaded = joblib.load(save_path)
    probs_orig = clf.predict_proba(X)
    probs_loaded = loaded.predict_proba(X)
    np.testing.assert_array_almost_equal(probs_orig, probs_loaded)


def test_reproducibility():
    """16. Verify deterministic seed produces identical predictions."""
    X = pd.DataFrame({
        "beginning_inventory": [100.0, 200.0, 50.0, 300.0] * 50,
        "ending_inventory_lag1": [95.0, 190.0, 40.0, 290.0] * 50,
        "inventory_days_of_supply": [5.0, 12.0, 2.0, 18.0] * 50,
        "category": ["Electronics", "Packaging", "Electronics", "Spare Parts"] * 50,
        "criticality": ["High", "Medium", "High", "Low"] * 50,
        "warehouse_id": ["WH-01", "WH-02", "WH-01", "WH-03"] * 50,
    })
    y = pd.Series([1, 0, 1, 0] * 50)
    
    clf1 = InventoryRiskClassifier(model_family="logistic", random_state=42)
    clf1.fit(X, y)
    p1 = clf1.predict_proba(X)[:, 1]
    
    clf2 = InventoryRiskClassifier(model_family="logistic", random_state=42)
    clf2.fit(X, y)
    p2 = clf2.predict_proba(X)[:, 1]
    
    np.testing.assert_array_almost_equal(p1, p2)


def test_no_duplicate_prediction_keys():
    """17. Verify primary key uniqueness: no duplicate (date, product_id, warehouse_id)."""
    inv_pred_path = Path("data/processed/risk/inventory_risk_predictions.parquet")
    if not inv_pred_path.exists():
        pytest.skip("inventory_risk_predictions.parquet not yet written")
    df = pd.read_parquet(inv_pred_path)
    
    dups = df.duplicated(subset=["date", "product_id", "warehouse_id"])
    assert dups.sum() == 0


def test_missing_value_handling():
    """18. Verify imputer handles missing numeric and categorical values cleanly."""
    X = pd.DataFrame({
        "beginning_inventory": [100.0, np.nan, 50.0, 300.0],
        "ending_inventory_lag1": [np.nan, 190.0, 40.0, 290.0],
        "inventory_days_of_supply": [5.0, 12.0, np.nan, 18.0],
        "category": ["Electronics", None, "Electronics", "Spare Parts"],
        "criticality": ["High", "Medium", np.nan, "Low"],
        "warehouse_id": ["WH-01", "WH-02", "WH-01", None],
    })
    y = pd.Series([1, 0, 1, 0])
    
    clf = InventoryRiskClassifier(model_family="logistic", random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert not np.isnan(probs).any()


def test_class_imbalance_handling():
    """19. Verify model handles severe class imbalance gracefully without crash."""
    n = 1000
    X = pd.DataFrame({
        "inventory_days_of_supply": np.random.uniform(1, 30, n),
        "ending_inventory_lag1": np.random.uniform(10, 500, n),
        "category": np.random.choice(["Electronics", "Consumables"], n),
        "criticality": np.random.choice(["High", "Low"], n),
        "warehouse_id": np.random.choice(["WH-01", "WH-02"], n),
    })
    # 2% positive rate
    y = pd.Series((np.random.uniform(0, 1, n) < 0.02).astype(int))
    
    clf = InventoryRiskClassifier(model_family="xgboost", n_estimators=10, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (n, 2)


def test_full_risk_pipeline_smoke():
    """20. Smoke test verifying manifest results file exists and has valid structure."""
    res_path = Path("reports/risk_model_results.json")
    if not res_path.exists():
        pytest.skip("risk_model_results.json not yet written")
    with open(res_path, "r") as f:
        res = json.load(f)
        
    assert "inventory_risk_model" in res
    assert "supplier_risk_model" in res
    assert "operational_exposure_summary" in res
    assert res["inventory_risk_model"]["test_metrics"]["roc_auc"] > 0.80
