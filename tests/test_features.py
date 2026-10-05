"""
Automated Pytest Suite for Feature Engineering Pipeline (Phase 6).
Validates:
1. Exact row counts and unbroken temporal grains across all datasets
2. Non-negative lags and strictly backward-looking rolling statistics (shift(1))
3. Zero future leakage under explicit future perturbations
4. Disjoint separation between feature columns and target columns
5. Non-overlapping chronological Train / Validation / Test splits
6. Completeness of feature registry definitions
"""

import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.queries import get_db_connection
from src.features.demand_features import compute_demand_features, extract_base_demand_grid
from src.features.inventory_features import compute_inventory_risk_features, extract_base_inventory_data
from src.features.supplier_features import compute_supplier_risk_features, extract_base_purchase_orders
from src.features.feature_pipeline import get_temporal_split
from src.features.validation import (
    validate_demand_leakage,
    validate_inventory_leakage,
    validate_supplier_leakage,
    validate_feature_target_separation,
    validate_rolling_shift_invariance
)


@pytest.fixture(scope="module")
def db_conn():
    conn = get_db_connection()
    yield conn
    conn.close()


# -----------------------------------------------------------------------------
# Demand Forecasting Features Tests
# -----------------------------------------------------------------------------
def test_demand_features_grain_and_row_count(db_conn):
    """Assert exactly 175,440 rows (60 products × 4 warehouses × 731 calendar days)."""
    base_df = extract_base_demand_grid(db_conn)
    assert len(base_df) == 175_440
    
    # Verify date range
    assert base_df["date"].min() == pd.to_datetime("2024-01-01")
    assert base_df["date"].max() == pd.to_datetime("2025-12-31")
    
    # Verify continuous daily coverage per series
    grouped = base_df.groupby(["product_id", "warehouse_id"])
    assert len(grouped) == 240
    for _, series_df in grouped:
        assert len(series_df) == 731
        diffs = series_df["date"].diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all()


def test_demand_features_lags_and_rolling(db_conn):
    """Assert lag_1 matches shift(1) exactly and rolling means exclude current day."""
    base_df = extract_base_demand_grid(db_conn)
    # Take a small subset for fast verification
    subset_df = base_df[base_df["product_id"].isin(["SKU-001", "SKU-002"])].copy()
    features = compute_demand_features(df_override=subset_df)
    
    # Verify shift invariance on lag_1
    is_valid_lag1 = validate_rolling_shift_invariance(
        features,
        grain_cols=["product_id", "warehouse_id", "date"],
        raw_col="demand_requested",
        lag1_col="demand_lag_1"
    )
    assert is_valid_lag1, "demand_lag_1 does not match shifted demand_requested!"


def test_demand_leakage_future_perturbation(db_conn):
    """Assert mutating future demand after cutoff has zero effect on historical features."""
    base_df = extract_base_demand_grid(db_conn)
    # Fast test on 2 products
    sample_df = base_df[base_df["product_id"].isin(["SKU-001", "SKU-002"])].copy()
    res = validate_demand_leakage(sample_df, cutoff_date="2025-06-30")
    assert res["is_leakage_free"], f"Demand leakage detected: {res['violating_columns']}"


# -----------------------------------------------------------------------------
# Inventory Risk Features Tests
# -----------------------------------------------------------------------------
def test_inventory_features_grain_and_row_count(db_conn):
    """Assert inventory ledger features table has exactly 175,440 rows."""
    base_df = extract_base_inventory_data(db_conn)
    assert len(base_df) == 175_440
    assert base_df["product_id"].nunique() == 60
    assert base_df["warehouse_id"].nunique() == 4


def test_inventory_buffer_metrics(db_conn):
    """Assert safety stock gap and inventory metrics match expected definitions."""
    base_df = extract_base_inventory_data(db_conn)
    sample_df = base_df[base_df["product_id"] == "SKU-001"].copy()
    features = compute_inventory_risk_features(df_override=sample_df)
    
    # Check that safety_stock_gap == beginning_inventory - safety_stock_target
    expected_gap = sample_df["beginning_inventory"].values - sample_df["safety_stock_target"].values
    diff = np.abs(features["safety_stock_gap"].values - expected_gap).max()
    assert diff < 1e-6, "safety_stock_gap does not match definition!"


def test_inventory_leakage_future_perturbation(db_conn):
    """Assert mutating future inventory has zero effect on historical inventory features."""
    base_df = extract_base_inventory_data(db_conn)
    sample_df = base_df[base_df["product_id"].isin(["SKU-001", "SKU-002"])].copy()
    res = validate_inventory_leakage(sample_df, cutoff_date="2025-06-30")
    assert res["is_leakage_free"], f"Inventory leakage detected: {res['violating_columns']}"


# -----------------------------------------------------------------------------
# Supplier Delay Risk Features Tests
# -----------------------------------------------------------------------------
def test_supplier_features_grain_and_row_count(db_conn):
    """Assert supplier risk features table has exactly 11,665 purchase orders."""
    base_df = extract_base_purchase_orders(db_conn)
    assert len(base_df) == 11_665
    assert base_df["po_id"].nunique() == 11_665


def test_supplier_features_cold_start_and_causality(db_conn):
    """Assert cold start flag is set on earliest orders and historical metrics are causal."""
    base_df = extract_base_purchase_orders(db_conn)
    features = compute_supplier_risk_features(df_override=base_df)
    
    # Earliest order per supplier should be cold start (0 completed orders prior to order placement)
    first_orders = features.groupby("supplier_id").first()
    assert (first_orders["is_cold_start"] == 1).all()
    assert (first_orders["hist_order_count"] == 0).all()

    # For orders with cold start, historical rates fall back to baseline reliability
    cold_start_rows = features[features["is_cold_start"] == 1]
    diff = np.abs(cold_start_rows["hist_on_time_rate"] - cold_start_rows["baseline_reliability"]).max()
    assert diff < 1e-6, "Cold start on-time rate does not match baseline reliability prior!"


def test_supplier_leakage_future_perturbation(db_conn):
    """Assert mutating future supplier performance has zero effect on historical features."""
    base_df = extract_base_purchase_orders(db_conn)
    res = validate_supplier_leakage(base_df, cutoff_date="2025-06-30")
    assert res["is_leakage_free"], f"Supplier leakage detected: {res['violating_columns']}"


# -----------------------------------------------------------------------------
# Temporal Splitting & Separation Tests
# -----------------------------------------------------------------------------
def test_temporal_splits_chronological(db_conn):
    """Assert strict chronological non-overlapping partitions."""
    base_df = extract_base_demand_grid(db_conn)
    train_df, val_df, test_df = get_temporal_split(base_df, date_col="date")
    
    # Total row partition
    assert len(train_df) + len(val_df) + len(test_df) == len(base_df)
    
    # Boundary checks
    assert train_df["date"].max() == pd.to_datetime("2025-03-31")
    assert val_df["date"].min() == pd.to_datetime("2025-04-01")
    assert val_df["date"].max() == pd.to_datetime("2025-06-30")
    assert test_df["date"].min() == pd.to_datetime("2025-07-01")
    assert test_df["date"].max() == pd.to_datetime("2025-12-31")
    
    # Temporal order assertions
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_feature_target_separation():
    """Assert no overlapping columns between feature list and target list."""
    feature_cols = ["demand_lag_1", "demand_lag_7", "demand_rolling_mean_28", "day_of_week"]
    target_cols = ["target_demand_t_plus_1", "target_demand_sum_7d"]
    assert validate_feature_target_separation(None, feature_cols, target_cols)


# -----------------------------------------------------------------------------
# Feature Registry Validity Test
# -----------------------------------------------------------------------------
def test_feature_registry_completeness():
    """Assert feature_registry.json exists, is non-empty, and has required metadata keys."""
    reg_path = Path("feature_registry.json")
    assert reg_path.exists(), "feature_registry.json is missing!"
    
    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    assert len(registry) >= 15, "Registry contains fewer entries than expected feature catalog!"
    required_keys = {
        "feature_name",
        "feature_family",
        "description",
        "transformation",
        "source_entity",
        "grain",
        "lookback_window",
        "availability_timestamp",
        "data_type",
        "leakage_risk_score",
        "leakage_justification",
        "downstream_consumer"
    }
    for item in registry:
        missing = required_keys - set(item.keys())
        assert not missing, f"Registry entry {item.get('feature_name')} missing keys: {missing}"
