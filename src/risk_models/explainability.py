"""
Deterministic Evidence-Based Explainability Engine for PPOI (Phase 8).
Generates auditable, feature-grounded explanations and recommended operational actions
for high-exposure SKU-Warehouse-Supplier positions without LLM hallucination.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_deterministic_evidence(row: pd.Series) -> List[str]:
    """
    Evaluates factual numerical thresholds on operational attributes to produce
    a list of deterministic evidence bullet points.
    """
    evidence: List[str] = []

    # 1. Demand vs Inventory Buffer Coverage
    fc_dem = float(row.get("forecast_demand_7d", 0.0))
    inv_pos = float(row.get("ending_inventory_lag1", row.get("beginning_inventory", 0.0)))
    dos = float(row.get("inventory_days_of_supply", 999.0))
    
    if fc_dem > inv_pos:
        deficit = fc_dem - inv_pos
        evidence.append(
            f"Projected 7-day demand ({fc_dem:.0f} units) exceeds available inventory ({inv_pos:.0f} units) by {deficit:.0f} units."
        )
    if dos < 5.0:
        evidence.append(f"Severely depleted inventory buffer: only {dos:.1f} days of supply remaining.")
    elif dos < 7.0:
        evidence.append(f"Sub-optimal inventory buffer: {dos:.1f} days of supply remaining (below 7-day planning target).")

    # 2. Safety Stock Deficit
    ss_target = float(row.get("safety_stock_target", 0.0))
    if inv_pos < ss_target and ss_target > 0:
        ss_gap = ss_target - inv_pos
        gap_pct = (ss_gap / ss_target) * 100.0
        evidence.append(
            f"Physical buffer below safety stock target ({inv_pos:.0f} vs {ss_target:.0f} units, {gap_pct:.1f}% deficit)."
        )

    # 3. Historical Stockout Recurrence
    stk_count = int(row.get("stockout_count_7d", 0))
    stk_rate = float(row.get("stockout_rate_28d", 0.0))
    if stk_count >= 2:
        evidence.append(f"Elevated recent stockout frequency: {stk_count} stockout days in the past 7 days.")
    elif stk_rate >= 0.15:
        evidence.append(f"Persistent historical stockout incidence: {stk_rate * 100:.1f}% of past 28 days experienced stockouts.")

    # 4. Supplier Delivery & Delay Risk
    sup_prob = float(row.get("supplier_delay_probability", 0.0))
    sup_on_time = float(row.get("hist_on_time_rate", 1.0))
    p90_del = float(row.get("hist_p90_delay", 0.0))
    sup_name = str(row.get("supplier_name", row.get("primary_supplier_name", "Primary Supplier")))

    if sup_prob >= 0.55:
        evidence.append(
            f"Primary supplier [{sup_name}] presents elevated delivery delay risk (predicted late probability: {sup_prob * 100:.1f}%)."
        )
    elif sup_on_time < 0.60:
        evidence.append(
            f"Supplier [{sup_name}] exhibits low historical reliability ({sup_on_time * 100:.1f}% on-time delivery rate)."
        )
    if p90_del >= 3.0:
        evidence.append(f"Severe tail-risk delivery delays: historical 90th percentile delay is {p90_del:.1f} days.")

    # 5. Supplier Capacity & Drift
    cap_press = float(row.get("supplier_capacity_pressure", 0.0))
    if cap_press >= 0.80:
        evidence.append(f"High supplier capacity pressure: active order volume is {cap_press * 100:.1f}% of monthly capacity.")

    # 6. Sourcing Vulnerability & Criticality
    crit = str(row.get("criticality", "Medium"))
    dep = str(row.get("supplier_dependency", ""))
    if crit.lower() == "high":
        evidence.append("High SKU operational criticality: stockout imposes immediate revenue and SLA penalties.")
    if "single" in dep.lower():
        evidence.append("Single-sourced dependency: no immediate secondary supplier alternative available on record.")

    # 7. Warehouse Utilization Pressure
    wh_util = float(row.get("warehouse_capacity_utilization_lag1", 0.0))
    if wh_util >= 85.0:
        evidence.append(f"Warehouse facility under severe saturation ({wh_util:.1f}% capacity utilization).")

    if not evidence:
        evidence.append("Stable buffer health and normal supplier delivery indicators under current operating parameters.")

    return evidence


def generate_recommended_action(row: pd.Series, risk_band: str) -> str:
    """
    Generates deterministic operational recommendations based on exposure band and feature drivers.
    """
    dos = float(row.get("inventory_days_of_supply", 999.0))
    sup_prob = float(row.get("supplier_delay_probability", 0.0))
    dep = str(row.get("supplier_dependency", "")).lower()

    if risk_band == "CRITICAL":
        if sup_prob >= 0.60 and "dual" in dep:
            return "URGENT: Expedite purchase order allocation to qualified secondary supplier; trigger emergency regional cross-dock transfer."
        elif dos < 4.0:
            return "CRITICAL: Issue emergency replenishment PO with express transit; evaluate temporary demand throttling or substitution."
        else:
            return "CRITICAL: Place urgent expedited PO and reserve safety stock buffer at regional hub."
    elif risk_band == "HIGH":
        if sup_prob >= 0.50:
            return "HIGH PRIORITY: Advance replenishment reorder date by 3 days to absorb expected supplier delay; audit supplier active pipeline."
        else:
            return "HIGH PRIORITY: Increase replenishment order quantity by safety stock deficit and monitor buffer daily."
    elif risk_band == "MEDIUM":
        return "ROUTINE REVIEW: Flag for standard procurement review during next weekly planning cycle; verify supplier lead-time adherence."
    else:
        return "MONITOR: Inventory and supply parameters within standard operating tolerances. No immediate intervention required."


def explain_operational_risk(row: pd.Series, risk_band: str) -> Tuple[str, List[str], str]:
    """
    Produces a composite explanation tuple:
        (concise_summary, evidence_list, recommended_action)
    """
    evidence = generate_deterministic_evidence(row)
    action = generate_recommended_action(row, risk_band)
    
    # Synthesize concise explanation
    if risk_band in ["HIGH", "CRITICAL"]:
        prefix = f"Elevated {risk_band.lower()} operational exposure because "
        details = "; ".join([e.rstrip(".") for e in evidence[:3]]) + "."
        summary = prefix + details[0].lower() + details[1:]
    else:
        summary = f"{risk_band} operational exposure: {evidence[0]}"
        
    return summary, evidence, action
