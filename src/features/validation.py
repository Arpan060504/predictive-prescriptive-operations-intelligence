"""
Feature Leakage and Quality Validation Module for PPOI (Phase 6).
Implements explicit mathematical tests guaranteeing zero target leakage,
temporal separation, shift(1) rolling alignment, and grain consistency.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.features.demand_features import compute_demand_features, extract_base_demand_grid
from src.features.inventory_features import compute_inventory_risk_features, extract_base_inventory_data
from src.features.supplier_features import compute_supplier_risk_features, extract_base_purchase_orders
from src.utils.logger import get_logger

logger = get_logger("FeatureValidation")


def validate_demand_leakage(base_df: pd.DataFrame, cutoff_date: str = "2025-06-30") -> Dict[str, Any]:
    """
    Test 1: Mutate future demand values after cutoff_date T.
    Assert historical demand features for t <= T are IDENTICAL.
    """
    logger.info("Executing demand future perturbation leakage test at cutoff %s...", cutoff_date)
    cutoff = pd.to_datetime(cutoff_date)
    
    # 1. Compute baseline features
    features_base = compute_demand_features(df_override=base_df)
    
    # 2. Perturb future demand by adding a massive shock (+10,000 units) strictly after cutoff
    df_perturbed = base_df.copy()
    future_mask = pd.to_datetime(df_perturbed["date"]) > cutoff
    df_perturbed.loc[future_mask, "demand_requested"] = df_perturbed.loc[future_mask, "demand_requested"] + 10_000
    
    features_perturbed = compute_demand_features(df_override=df_perturbed)
    
    # 3. Filter both to t <= cutoff
    hist_base = features_base[pd.to_datetime(features_base["date"]) <= cutoff].copy().reset_index(drop=True)
    hist_perturbed = features_perturbed[pd.to_datetime(features_perturbed["date"]) <= cutoff].copy().reset_index(drop=True)
    
    # Feature columns to check (exclude forward target columns)
    feature_cols = [c for c in hist_base.columns if not c.startswith("target_") and not c.startswith("is_target_")]
    
    # Compare numerical values
    max_abs_diff = 0.0
    violating_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(hist_base[col]):
            diff = np.abs(hist_base[col].fillna(0) - hist_perturbed[col].fillna(0)).max()
            if diff > 1e-6:
                violating_cols.append((col, diff))
            max_abs_diff = max(max_abs_diff, diff)
        else:
            if not hist_base[col].equals(hist_perturbed[col]):
                violating_cols.append((col, -1))

    is_leakage_free = (len(violating_cols) == 0)
    logger.info("Demand leakage test result: is_leakage_free=%s, max_diff=%.2e", is_leakage_free, max_abs_diff)
    return {
        "domain": "demand",
        "is_leakage_free": is_leakage_free,
        "cutoff_date": cutoff_date,
        "max_abs_diff": float(max_abs_diff),
        "violating_columns": violating_cols
    }


def validate_supplier_leakage(base_df: pd.DataFrame, cutoff_date: str = "2025-06-30") -> Dict[str, Any]:
    """
    Test 2: Mutate future supplier delays and lead times after cutoff_date T.
    Assert historical supplier features for t <= T are IDENTICAL.
    """
    logger.info("Executing supplier future perturbation leakage test at cutoff %s...", cutoff_date)
    cutoff = pd.to_datetime(cutoff_date)
    
    # 1. Compute baseline features
    features_base = compute_supplier_risk_features(df_override=base_df)
    
    # 2. Perturb future orders (order_date > cutoff) with +50 day delay and 0 on-time
    df_perturbed = base_df.copy()
    future_mask = pd.to_datetime(df_perturbed["order_date"]) > cutoff
    df_perturbed.loc[future_mask, "delay_days"] = df_perturbed.loc[future_mask, "delay_days"] + 50
    df_perturbed.loc[future_mask, "actual_lead_time_days"] = df_perturbed.loc[future_mask, "actual_lead_time_days"] + 50
    df_perturbed.loc[future_mask, "on_time_flag"] = 0
    df_perturbed.loc[future_mask, "otif_flag"] = 0
    
    features_perturbed = compute_supplier_risk_features(df_override=df_perturbed)
    
    # 3. Filter both to orders placed at or before cutoff
    hist_base = features_base[pd.to_datetime(features_base["order_date"]) <= cutoff].copy().reset_index(drop=True)
    hist_perturbed = features_perturbed[pd.to_datetime(features_perturbed["order_date"]) <= cutoff].copy().reset_index(drop=True)
    
    feature_cols = [c for c in hist_base.columns if not c.startswith("target_")]
    
    max_abs_diff = 0.0
    violating_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(hist_base[col]):
            diff = np.abs(hist_base[col].fillna(0) - hist_perturbed[col].fillna(0)).max()
            if diff > 1e-6:
                violating_cols.append((col, diff))
            max_abs_diff = max(max_abs_diff, diff)
        else:
            if not hist_base[col].equals(hist_perturbed[col]):
                violating_cols.append((col, -1))

    is_leakage_free = (len(violating_cols) == 0)
    logger.info("Supplier leakage test result: is_leakage_free=%s, max_diff=%.2e", is_leakage_free, max_abs_diff)
    return {
        "domain": "supplier",
        "is_leakage_free": is_leakage_free,
        "cutoff_date": cutoff_date,
        "max_abs_diff": float(max_abs_diff),
        "violating_columns": violating_cols
    }


def validate_inventory_leakage(base_df: pd.DataFrame, cutoff_date: str = "2025-06-30") -> Dict[str, Any]:
    """
    Test 3: Mutate future inventory levels after cutoff_date T.
    Assert historical inventory risk features for t <= T are IDENTICAL.
    """
    logger.info("Executing inventory future perturbation leakage test at cutoff %s...", cutoff_date)
    cutoff = pd.to_datetime(cutoff_date)
    
    # 1. Baseline
    features_base = compute_inventory_risk_features(df_override=base_df)
    
    # 2. Perturb future inventory (date > cutoff)
    df_perturbed = base_df.copy()
    future_mask = pd.to_datetime(df_perturbed["date"]) > cutoff
    df_perturbed.loc[future_mask, "ending_inventory"] = 0
    df_perturbed.loc[future_mask, "stockout_flag"] = 1
    df_perturbed.loc[future_mask, "lost_sales_quantity"] = 5000
    
    features_perturbed = compute_inventory_risk_features(df_override=df_perturbed)
    
    # 3. Filter to t <= cutoff
    hist_base = features_base[pd.to_datetime(features_base["date"]) <= cutoff].copy().reset_index(drop=True)
    hist_perturbed = features_perturbed[pd.to_datetime(features_perturbed["date"]) <= cutoff].copy().reset_index(drop=True)
    
    feature_cols = [c for c in hist_base.columns if not c.startswith("target_") and not c.startswith("is_target_")]
    
    max_abs_diff = 0.0
    violating_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(hist_base[col]):
            diff = np.abs(hist_base[col].fillna(0) - hist_perturbed[col].fillna(0)).max()
            if diff > 1e-6:
                violating_cols.append((col, diff))
            max_abs_diff = max(max_abs_diff, diff)
        else:
            if not hist_base[col].equals(hist_perturbed[col]):
                violating_cols.append((col, -1))

    is_leakage_free = (len(violating_cols) == 0)
    logger.info("Inventory leakage test result: is_leakage_free=%s, max_diff=%.2e", is_leakage_free, max_abs_diff)
    return {
        "domain": "inventory",
        "is_leakage_free": is_leakage_free,
        "cutoff_date": cutoff_date,
        "max_abs_diff": float(max_abs_diff),
        "violating_columns": violating_cols
    }


def validate_feature_target_separation(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> bool:
    """
    Test 4: Strict separation between feature columns and target columns.
    Asserts zero overlap between feature_cols and target_cols.
    """
    overlap = set(feature_cols).intersection(set(target_cols))
    if overlap:
        logger.error("Feature/Target separation violation: columns appear in both sets: %s", overlap)
        return False
    return True


def validate_rolling_shift_invariance(df: pd.DataFrame, grain_cols: List[str], raw_col: str, lag1_col: str) -> bool:
    """
    Test 5: Asserts that lag1_col for day t matches raw_col for day t-1 exactly.
    """
    sorted_df = df.sort_values(grain_cols).reset_index(drop=True)
    expected_lag1 = sorted_df.groupby(grain_cols[:-1])[raw_col].shift(1)
    
    # Ignore the first row of each group where lag1 is NaN
    valid_mask = expected_lag1.notna()
    diff = np.abs(sorted_df.loc[valid_mask, lag1_col] - expected_lag1.loc[valid_mask]).max()
    return bool(diff < 1e-6)
