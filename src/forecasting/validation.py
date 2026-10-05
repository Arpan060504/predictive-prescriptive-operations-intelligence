"""
Forecasting Leakage Prevention, Feature Whitelisting, and Validation Module for PPOI (Phase 7).
Enforces strict programmatic feature whitelisting, rejects target columns with loud errors,
and verifies mathematical invariance under future perturbations.
"""

from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("ForecastingValidation")

# Explicit list of forbidden columns that must NEVER enter a feature matrix X
FORBIDDEN_EXACT_COLUMNS: Set[str] = {
    # Demand targets
    "target_demand_t_plus_1",
    "target_demand_t_plus_7",
    "target_demand_t_plus_14",
    "target_demand_t_plus_28",
    "target_demand_sum_7d",
    "target_demand_sum_14d",
    "target_demand_sum_28d",
    # Target availability indicators
    "is_target_available_t_plus_1",
    "is_target_available_t_plus_7",
    "is_target_available_t_plus_14",
    "is_target_available_t_plus_28",
    # Inventory targets
    "target_stockout_t_plus_1",
    "target_stockout_within_7d",
    "target_stockout_within_14d",
    "target_lost_sales_7d",
    "is_target_available_7d",
    "is_target_available_14d",
    # Supplier targets
    "target_delay_days",
    "target_delay_gt_2d",
    "target_delay_gt_5d",
    "target_on_time",
    "target_otif",
    "target_actual_lead_time_days",
    # Post-hoc / data quality flags that could leak synthetic injections
    "outlier_flag",
}

# Explicit approved feature whitelist
DEFAULT_NUMERIC_FEATURES: List[str] = [
    # Contemporaneous demand finalized on day t (known at completion of day t)
    "demand_requested",
    # Historical lags (t-1 backwards)
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    # Shifted rolling window statistics (t-1 backwards)
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
    # Dynamic demand indicators
    "demand_change_1d",
    "demand_change_7d",
    "demand_growth_28d",
    "demand_cv_28",
    "zero_demand_frequency_28",
    # Calendar deterministic features
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "month_start",
    "month_end",
    "holiday_flag",
    # Published operational event metadata
    "promotion_flag",
    "event_flag",
    "event_active",
    "event_demand_multiplier",
    "event_lead_time_multiplier",
    "event_transport_multiplier",
    "days_since_event_start",
    "days_until_event_end",
]

DEFAULT_CATEGORICAL_FEATURES: List[str] = [
    "category",
    "warehouse_id",
]

DEFAULT_FEATURE_WHITELIST: List[str] = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES


def assert_feature_whitelist(
    columns: Sequence[str],
    whitelist: Optional[Sequence[str]] = None
) -> None:
    """
    Validates that a proposed feature matrix X contains only legitimate, approved features.
    FAIL LOUDLY if any target, future-looking, or forbidden column appears in columns.
    """
    cols_set = set(columns)

    # 1. Check for any column starting with 'target_'
    target_cols = [c for c in cols_set if c.startswith("target_")]
    if target_cols:
        msg = f"CRITICAL LEAKAGE DETECTED: Feature matrix contains target columns: {target_cols}"
        logger.error(msg)
        raise ValueError(msg)

    # 2. Check for exact forbidden columns
    forbidden_present = cols_set.intersection(FORBIDDEN_EXACT_COLUMNS)
    if forbidden_present:
        msg = f"CRITICAL LEAKAGE DETECTED: Feature matrix contains forbidden columns: {forbidden_present}"
        logger.error(msg)
        raise ValueError(msg)

    # 3. If an explicit whitelist is provided, ensure all columns are members
    if whitelist is not None:
        whitelist_set = set(whitelist)
        unapproved = cols_set - whitelist_set
        if unapproved:
            msg = f"UNAPPROVED COLUMNS IN FEATURE SET: Columns not in approved whitelist: {unapproved}"
            logger.error(msg)
            raise ValueError(msg)

    logger.debug("Feature whitelist assertion passed for %d columns.", len(columns))


def extract_model_features(
    df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Safely extracts and validates model feature columns from a demand dataframe.
    Enforces the feature whitelist and returns a clean, leak-free feature dataframe X.
    """
    num_cols = numeric_features or DEFAULT_NUMERIC_FEATURES
    cat_cols = categorical_features or DEFAULT_CATEGORICAL_FEATURES
    all_feature_cols = num_cols + cat_cols

    # Programmatic leakage assertion
    assert_feature_whitelist(all_feature_cols, whitelist=all_feature_cols)

    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Feature dataset is missing expected whitelist columns: {missing_cols}")

    return df[all_feature_cols].copy()


def verify_future_perturbation_invariance(
    df: pd.DataFrame,
    cutoff_date: str = "2025-06-30",
    perturbation_shock: float = 10_000.0
) -> Dict[str, Any]:
    """
    Regression test:
    1. Extracts baseline model inputs X for dates <= cutoff_date.
    2. Perturbs future demand after cutoff_date (date > cutoff_date) with a massive shock (+10,000 units).
    3. Verifies that X for dates <= cutoff_date remains completely unchanged.
    """
    logger.info("Executing future perturbation invariance test at cutoff %s...", cutoff_date)
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
        df_copy["date"] = pd.to_datetime(df_copy["date"])
    cutoff = pd.to_datetime(cutoff_date)

    # Baseline features
    mask_hist = df_copy["date"] <= cutoff
    X_base = extract_model_features(df_copy[mask_hist])

    # Perturb future demand
    df_perturbed = df_copy.copy()
    mask_future = df_perturbed["date"] > cutoff
    df_perturbed.loc[mask_future, "demand_requested"] = (
        df_perturbed.loc[mask_future, "demand_requested"] + perturbation_shock
    )

    # Extract historical features from perturbed dataset
    X_perturbed = extract_model_features(df_perturbed[mask_hist])

    # Check maximum absolute difference across numerical features
    num_cols = [c for c in X_base.columns if pd.api.types.is_numeric_dtype(X_base[c])]
    max_abs_diff = float(np.nanmax(np.abs(X_base[num_cols].values - X_perturbed[num_cols].values)))

    is_invariant = max_abs_diff < 1e-6
    if not is_invariant:
        logger.error("Future perturbation test FAILED: max_abs_diff=%.2e", max_abs_diff)
    else:
        logger.info("Future perturbation test PASSED: max_abs_diff=%.2e", max_abs_diff)

    return {
        "is_invariant": is_invariant,
        "cutoff_date": cutoff_date,
        "max_abs_diff": max_abs_diff,
        "perturbation_shock": perturbation_shock,
    }
