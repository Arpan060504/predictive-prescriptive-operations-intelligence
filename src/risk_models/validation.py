"""
Risk Feature Validation & Leakage Prevention Module for PPOI (Phase 8).
Enforces programmatic feature whitelists and future perturbation invariance tests
to guarantee that no target variables or future observations leak into risk models.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("RiskValidation")

# -----------------------------------------------------------------------------
# Certified Inventory Feature Whitelist (Available at decision timestamp t)
# -----------------------------------------------------------------------------
INVENTORY_FEATURE_WHITELIST: List[str] = [
    # Physical Buffer & Inventory State (Lagged or known before fulfillment)
    "beginning_inventory",
    "ending_inventory_lag1",
    "ending_inventory_lag2",
    "inventory_change_1d",
    "safety_stock_target",
    "safety_stock_gap",
    "inventory_to_safety_stock_ratio",
    # Historical Demand & Volatility (Shifted to t-1)
    "demand_mean_7",
    "demand_mean_28",
    "demand_std_28",
    "demand_trend_28",
    "inventory_days_of_supply",
    # Historical Stockout & Loss Incidence
    "stockout_lag_1",
    "stockout_count_7d",
    "stockout_count_28d",
    "stockout_rate_28d",
    "consecutive_stockout_days",
    "lost_sales_7d",
    "lost_sales_28d",
    "lost_sales_rate_28d",
    # Backorder Dynamics
    "ending_backorder_lag1",
    "backorder_count_7d",
    "backorder_count_28d",
    # Warehouse Saturation Context
    "warehouse_total_inv_lag1",
    "warehouse_capacity_utilization_lag1",
    "warehouse_utilization_7d",
    "warehouse_inventory_share",
    # Product Dimension Economics
    "unit_cost",
    "selling_price",
    # Forward Forecast Signal (From locked Phase 7 forecast engine)
    "forecast_demand_7d",
    # Categoricals / Dimensions
    "category",
    "criticality",
    "warehouse_id",
]

# Explicitly forbidden columns for Inventory Risk modeling
FORBIDDEN_INVENTORY_COLUMNS: List[str] = [
    # Contemporaneous day t outcome variables
    "demand_fulfilled",
    "lost_sales_quantity",
    "ending_backorder",
    "ending_inventory",
    "stockout_flag",
    "holding_cost",
    "stockout_cost",
    "po_received",
    # Target variables
    "target_stockout_t_plus_1",
    "target_stockout_within_7d",
    "target_stockout_within_14d",
    "target_lost_sales_7d",
]

# -----------------------------------------------------------------------------
# Certified Supplier Feature Whitelist (Available strictly prior to PO order date)
# -----------------------------------------------------------------------------
SUPPLIER_FEATURE_WHITELIST: List[str] = [
    # Static Supplier Prior Dimensions
    "baseline_reliability",
    "baseline_lead_time_days",
    "lead_time_variance",
    "monthly_capacity_units",
    "moq_units",
    "supplier_transport_cost_factor",
    # Point-in-Time Historical Track Record (Completed strictly before order_date)
    "hist_order_count",
    "hist_order_volume",
    "hist_on_time_rate",
    "hist_otif_rate",
    "hist_fill_rate",
    "hist_mean_delay",
    "hist_std_delay",
    "hist_p90_delay",
    "hist_p95_delay",
    # Recent Drift & Momentum
    "recent_delay_30d",
    "recent_otif_30d",
    "recent_otif_90d",
    "delay_trend",
    "days_since_last_delivery",
    "is_cold_start",
    # Capacity Pressure
    "supplier_active_orders_count",
    "supplier_active_volume",
    "supplier_capacity_pressure",
    # Order Context at Placement
    "quantity_ordered",
    "expected_lead_time_days",
    "unit_cost",
    "order_day_of_week",
    "order_month",
    "order_quarter",
    "order_is_weekend",
    "order_is_month_end",
    "hist_sku_unit_cost_mean",
    "hist_sku_unit_cost_std",
    # Categoricals / Dimensions
    "supplier_id",
    "supplier_tier",
    "warehouse_id",
]

# Explicitly forbidden columns for Supplier Risk modeling
FORBIDDEN_SUPPLIER_COLUMNS: List[str] = [
    # PO Execution Outcomes
    "actual_lead_time_days",
    "actual_delivery_date",
    "quantity_received",
    "total_procurement_cost",
    "transport_cost",
    "delay_days",
    "on_time_flag",
    "in_full_flag",
    "otif_flag",
    "status",
    # Targets
    "target_delay_days",
    "target_delay_gt_2d",
    "target_delay_gt_5d",
    "target_on_time",
    "target_otif",
    "target_actual_lead_time_days",
]


def assert_risk_feature_whitelist(
    df: pd.DataFrame,
    whitelist: Sequence[str],
    forbidden_exact: Sequence[str],
    forbidden_prefixes: Sequence[str] = ("target_",),
    context_name: str = "RiskModel",
) -> None:
    """
    Validates that a feature DataFrame contains only approved whitelisted columns
    and raises an immediate ValueError if any forbidden target or future column is present.
    """
    df_cols = set(df.columns)
    
    # 1. Check for exact forbidden columns
    violating_exact = [col for col in forbidden_exact if col in df_cols]
    if violating_exact:
        raise ValueError(
            f"[{context_name}] FATAL LEAKAGE DETECTED: Features contain forbidden outcome columns: {violating_exact}"
        )
        
    # 2. Check for forbidden prefixes (e.g. target_*)
    violating_prefixes = [
        col for col in df_cols
        if any(col.startswith(pfx) for pfx in forbidden_prefixes)
    ]
    if violating_prefixes:
        raise ValueError(
            f"[{context_name}] FATAL LEAKAGE DETECTED: Features contain target-prefixed columns: {violating_prefixes}"
        )
        
    # 3. Check for unwhitelisted columns
    unapproved = [col for col in df_cols if col not in whitelist]
    if unapproved:
        raise ValueError(
            f"[{context_name}] UNAPPROVED FEATURES DETECTED: The following columns are not on the certified whitelist: {unapproved}"
        )
        
    logger.debug("[%s] Feature whitelist passed successfully (%d columns).", context_name, len(df_cols))


def verify_future_perturbation_invariance(
    feature_builder_fn: Callable[[pd.DataFrame], pd.DataFrame],
    base_data: pd.DataFrame,
    split_date: str = "2025-06-30",
    perturb_col: str = "demand_requested",
    perturb_delta: float = 50000.0,
) -> float:
    """
    Empirical Future Perturbation Regression Test.
    
    Mutates future observations strictly after `split_date` and re-runs the feature
    builder. Verifies that historical feature inputs for dates <= `split_date` remain
    100.00% identical.
    
    Returns the maximum absolute difference across all historical feature cells.
    """
    logger.info("Executing future perturbation invariance test across split_date=%s...", split_date)
    
    # 1. Base run
    df_base = base_data.copy()
    features_base = feature_builder_fn(df_base)
    
    # 2. Perturb future observations
    df_perturbed = base_data.copy()
    if "date" in df_perturbed.columns:
        date_mask = pd.to_datetime(df_perturbed["date"]) > pd.to_datetime(split_date)
    elif "order_date" in df_perturbed.columns:
        date_mask = pd.to_datetime(df_perturbed["order_date"]) > pd.to_datetime(split_date)
    else:
        raise ValueError("Neither 'date' nor 'order_date' column found in data.")
        
    if perturb_col in df_perturbed.columns:
        df_perturbed.loc[date_mask, perturb_col] = df_perturbed.loc[date_mask, perturb_col] + perturb_delta
        
    features_perturbed = feature_builder_fn(df_perturbed)
    
    # 3. Filter to historical period <= split_date
    if "date" in features_base.columns:
        hist_mask = pd.to_datetime(features_base["date"]) <= pd.to_datetime(split_date)
    else:
        hist_mask = pd.to_datetime(features_base["order_date"]) <= pd.to_datetime(split_date)
        
    numeric_cols = features_base.select_dtypes(include=[np.number]).columns
    
    # Exclude targets if present in raw feature build
    check_cols = [c for c in numeric_cols if not c.startswith("target_") and not c.startswith("is_target_")]
    
    diff = np.abs(
        features_base.loc[hist_mask, check_cols].values - 
        features_perturbed.loc[hist_mask, check_cols].values
    )
    max_diff = float(np.nanmax(diff))
    
    logger.info("Future perturbation max absolute difference on historical inputs: %.4e", max_diff)
    if max_diff > 1e-6:
        raise ValueError(
            f"LEAKAGE TEST FAILED: Perturbing future observations changed historical features by max_diff={max_diff:.4e}"
        )
        
    return max_diff
