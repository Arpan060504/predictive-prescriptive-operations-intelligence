"""
Policy Comparison and Trade-off Engine for PPOI Prescriptive Intelligence.

Performs rigorous comparative analytics between the Constrained Multi-Echelon Optimizer
and the Decentralized Heuristic Baseline Policy across cost components, service levels,
facility utilization, and sourcing concentration.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.optimization.optimizer import OptimizationResult
from src.optimization.baseline import BaselineResult


def compute_hhi(shares: np.ndarray) -> float:
    """
    Computes Herfindahl-Hirschman Index (HHI) for sourcing concentration.
    HHI ranges from ~0 (perfectly diversified) to 10,000 (pure monopoly / 100% single supplier).
    """
    if len(shares) == 0 or np.sum(shares) == 0:
        return 0.0
    pct = (shares / np.sum(shares)) * 100.0
    return float(np.sum(pct ** 2))


def compare_policies(
    opt_res: OptimizationResult,
    base_res: BaselineResult,
) -> Dict[str, Any]:
    """
    Compares the Prescriptive Optimizer against the Baseline Heuristic.
    
    Returns:
        Structured dictionary containing metrics, cost waterfall, and sourcing comparison.
    """
    cost_diff = base_res.total_cost - opt_res.total_cost
    cost_reduction_pct = (cost_diff / base_res.total_cost * 100.0) if base_res.total_cost > 0 else 0.0

    # 1. Cost Waterfall Breakdown
    categories = [
        "procurement_cost",
        "transport_cost",
        "transshipment_cost",
        "holding_cost",
        "shortage_cost",
        "supplier_risk_cost",
        "total_landed_cost",
    ]
    waterfall_rows = []
    for cat in categories:
        b_val = float(base_res.cost_breakdown.get(cat, 0.0))
        o_val = float(opt_res.cost_breakdown.get(cat, 0.0))
        diff = b_val - o_val
        pct = (diff / b_val * 100.0) if b_val > 0 else 0.0
        waterfall_rows.append({
            "cost_category": cat,
            "baseline_cost": round(b_val, 2),
            "optimized_cost": round(o_val, 2),
            "modeled_difference": round(diff, 2),
            "pct_change": round(pct, 2),
        })
    df_waterfall = pd.DataFrame(waterfall_rows)

    # 2. Supplier Sourcing Concentration (HHI)
    base_sup_totals = (
        base_res.orders.groupby("supplier_id")["order_quantity"].sum()
        if not base_res.orders.empty else pd.Series(dtype=float)
    )
    opt_sup_totals = (
        opt_res.orders.groupby("supplier_id")["order_quantity"].sum()
        if not opt_res.orders.empty else pd.Series(dtype=float)
    )

    base_hhi = compute_hhi(base_sup_totals.values)
    opt_hhi = compute_hhi(opt_sup_totals.values)

    # Sourcing shares breakdown
    all_sups = sorted(list(set(base_sup_totals.index.tolist() + opt_sup_totals.index.tolist())))
    sourcing_rows = []
    for s in all_sups:
        b_qty = float(base_sup_totals.get(s, 0.0))
        o_qty = float(opt_sup_totals.get(s, 0.0))
        b_share = (b_qty / base_res.total_ordered * 100.0) if base_res.total_ordered > 0 else 0.0
        o_share = (o_qty / opt_res.total_ordered * 100.0) if opt_res.total_ordered > 0 else 0.0
        sourcing_rows.append({
            "supplier_id": s,
            "baseline_quantity": round(b_qty, 2),
            "baseline_share_pct": round(b_share, 2),
            "optimized_quantity": round(o_qty, 2),
            "optimized_share_pct": round(o_share, 2),
        })
    df_sourcing = pd.DataFrame(sourcing_rows)

    # 3. Warehouse Inventory & Shortage Summary
    wh_summary_rows = []
    base_wh_short = (
        base_res.shortages.groupby("warehouse_id")["shortage_quantity"].sum()
        if not base_res.shortages.empty else pd.Series(dtype=float)
    )
    opt_wh_short = (
        opt_res.shortages.groupby("warehouse_id")["shortage_quantity"].sum()
        if not opt_res.shortages.empty else pd.Series(dtype=float)
    )
    base_wh_inv = (
        base_res.inventory.groupby("warehouse_id")["ending_inventory"].sum()
        if not base_res.inventory.empty else pd.Series(dtype=float)
    )
    opt_wh_inv = (
        opt_res.inventory.groupby("warehouse_id")["ending_inventory"].sum()
        if not opt_res.inventory.empty else pd.Series(dtype=float)
    )

    all_whs = sorted(list(set(base_wh_inv.index.tolist() + opt_wh_inv.index.tolist())))
    for w in all_whs:
        wh_summary_rows.append({
            "warehouse_id": w,
            "baseline_ending_inv": round(float(base_wh_inv.get(w, 0.0)), 2),
            "optimized_ending_inv": round(float(opt_wh_inv.get(w, 0.0)), 2),
            "baseline_shortage": round(float(base_wh_short.get(w, 0.0)), 2),
            "optimized_shortage": round(float(opt_wh_short.get(w, 0.0)), 2),
        })
    df_wh_summary = pd.DataFrame(wh_summary_rows)

    return {
        "scenario_name": opt_res.scenario_name,
        "baseline_total_cost": base_res.total_cost,
        "optimized_total_cost": opt_res.total_cost,
        "cost_difference": round(cost_diff, 2),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
        "baseline_service_level": base_res.service_level,
        "optimized_service_level": opt_res.service_level,
        "baseline_hhi": round(base_hhi, 1),
        "optimized_hhi": round(opt_hhi, 1),
        "total_transfers_units": opt_res.total_transferred,
        "cost_waterfall": df_waterfall,
        "sourcing_breakdown": df_sourcing,
        "warehouse_summary": df_wh_summary,
    }
