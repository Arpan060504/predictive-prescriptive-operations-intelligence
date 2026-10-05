"""
Integrated Operational Exposure Engine for PPOI (Phase 8).
Combines multi-horizon demand forecasts, inventory buffer health,
and supplier reliability into prioritized, auditable operational exposure scores.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.risk_models.explainability import explain_operational_risk
from src.utils.logger import get_logger

logger = get_logger("OperationalExposure")


def build_operational_exposure_layer(
    inventory_risk_df: pd.DataFrame,
    supplier_risk_profile_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    dim_supplier_df: pd.DataFrame,
    stockout_penalty_multiplier: float = 1.5,
    exposure_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Constructs the integrated SKU × Warehouse × Supplier operational exposure layer.
    
    Mathematical Formulation:
        Exposure Score = w_inv * S_inv + w_sup * S_sup + w_inter * ((S_inv * S_sup) / 100)
        Projected Exposure Quantity = max(0, Forecast_7d - Inventory_Position) * P_stockout
        Projected Exposure Cost = Exposure_Quantity * Unit_Cost * Stockout_Penalty_Multiplier
    """
    logger.info("Synthesizing integrated operational exposure layer (%d inventory records)...", len(inventory_risk_df))
    
    if exposure_weights is None:
        exposure_weights = {
            "w_inv": 0.45,
            "w_sup": 0.35,
            "w_inter": 0.20,
        }
    w_sum = sum(exposure_weights.values())
    w = {k: v / w_sum for k, v in exposure_weights.items()}

    # Merge product dimension attributes (primary supplier, unit cost, criticality, dependency)
    prod_cols = [
        "product_id", "primary_supplier_id", "secondary_supplier_id",
        "supplier_dependency", "shelf_life_days", "storage_requirement"
    ]
    prod_sub = dim_product_df[[c for c in prod_cols if c in dim_product_df.columns]].drop_duplicates()
    
    df = inventory_risk_df.merge(prod_sub, on="product_id", how="left")

    # Merge latest supplier risk profile by primary_supplier_id
    sup_cols = [
        "supplier_id", "supplier_name", "supplier_tier",
        "supplier_delay_probability", "supplier_risk_score", "supplier_risk_band",
        "hist_on_time_rate", "hist_p90_delay", "supplier_capacity_pressure", "delay_trend"
    ]
    sup_sub = supplier_risk_profile_df[[c for c in sup_cols if c in supplier_risk_profile_df.columns]].drop_duplicates(subset=["supplier_id"])
    
    df = df.merge(
        sup_sub,
        left_on="primary_supplier_id",
        right_on="supplier_id",
        how="left"
    )

    # Defaults for unmatched suppliers if any
    df["supplier_delay_probability"] = df["supplier_delay_probability"].fillna(0.45)
    df["supplier_risk_score"] = df["supplier_risk_score"].fillna(45.0)
    df["supplier_risk_band"] = df["supplier_risk_band"].fillna("MEDIUM")

    # Compute operational exposure score
    s_inv = df["inventory_risk_score"].values
    s_sup = df["supplier_risk_score"].values
    s_inter = (s_inv * s_sup) / 100.0

    exposure_score = (
        w["w_inv"] * s_inv +
        w["w_sup"] * s_sup +
        w["w_inter"] * s_inter
    )
    df["operational_exposure_score"] = np.clip(np.round(exposure_score, 1), 0.0, 100.0)

    # Exposure bands: LOW [0, 25), MEDIUM [25, 55), HIGH [55, 75), CRITICAL [75, 100]
    exposure_bands = []
    for s in df["operational_exposure_score"]:
        if s < 25.0:
            exposure_bands.append("LOW")
        elif s < 55.0:
            exposure_bands.append("MEDIUM")
        elif s < 75.0:
            exposure_bands.append("HIGH")
        else:
            exposure_bands.append("CRITICAL")
    df["operational_exposure_band"] = exposure_bands

    # Projected exposure quantity and cost
    inv_pos = df["ending_inventory_lag1"].fillna(df["beginning_inventory"]).values
    fc_dem = df["forecast_demand_7d"].fillna(df["demand_mean_7"] * 7.0).values
    p_stk = df["stockout_probability_7d"].values
    unit_cost = df["unit_cost"].values

    deficit = np.maximum(0.0, fc_dem - inv_pos)
    exposure_qty = deficit * p_stk
    exposure_cost = exposure_qty * unit_cost * stockout_penalty_multiplier

    df["projected_exposure_quantity"] = np.round(exposure_qty, 1)
    df["projected_exposure_cost"] = np.round(exposure_cost, 2)

    logger.info("Generating deterministic explanations for %d positions...", len(df))
    summaries = []
    actions = []
    evidence_strings = []

    for _, row in df.iterrows():
        summary, ev_list, action = explain_operational_risk(row, row["operational_exposure_band"])
        summaries.append(summary)
        actions.append(action)
        evidence_strings.append(" | ".join(ev_list[:4]))

    df["explanation_summary"] = summaries
    df["recommended_action"] = actions
    df["evidence_bullets"] = evidence_strings

    logger.info("Operational exposure layer constructed successfully.")
    return df
