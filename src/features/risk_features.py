"""
Composite Operational Risk Feature Engineering Module for PPOI (Phase 6).
Computes cross-functional operational risk indices combining inventory buffer deficiency,
demand volatility, and supplier reliability for prescriptive optimization input.
"""

from typing import Optional
import numpy as np
import pandas as pd


def compute_buffer_deficiency_index(
    inventory_level: pd.Series,
    safety_stock: pd.Series
) -> pd.Series:
    """
    Computes normalized buffer deficiency index:
    (safety_stock - current_inventory) / safety_stock.
    Values > 0 indicate inventory below safety buffer; capped between [-2.0, 1.0].
    """
    safe_ss = np.maximum(safety_stock, 1.0)
    deficiency = (safe_ss - inventory_level) / safe_ss
    return np.clip(deficiency, -2.0, 1.0)


def compute_supplier_vulnerability_score(
    supplier_delay_prob: pd.Series,
    product_criticality: pd.Series
) -> pd.Series:
    """
    Computes supplier vulnerability score weighting delay probability
    by product criticality weight (Critical=3.0, High=2.0, Medium=1.5, Low=1.0).
    """
    criticality_weights = {
        "CRITICAL": 3.0,
        "HIGH": 2.0,
        "MEDIUM": 1.5,
        "LOW": 1.0
    }
    crit_weight = product_criticality.str.upper().map(criticality_weights).fillna(1.5)
    return supplier_delay_prob * crit_weight


def compute_expected_stockout_exposure(
    stockout_prob: pd.Series,
    stockout_cost_per_unit: pd.Series,
    expected_daily_demand: pd.Series
) -> pd.Series:
    """
    Computes expected financial exposure from stockout:
    exposure = stockout_probability * daily_demand * stockout_cost_per_unit.
    """
    return stockout_prob * expected_daily_demand * stockout_cost_per_unit
