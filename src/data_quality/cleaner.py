"""
Data Cleaning and Remediation Engine for PPOI.
Performs transparent, deterministic remediation on corrupted operational data.
Logs every single repair, quarantine, and deduplication action into an explicit audit trail.
Saves the certified clean analytical dataset to data/processed/.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("DataCleaner")


def clean_operational_dataset(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Cleans all operational tables from input_dir and saves certified clean tables to output_dir.
    Returns:
    - cleaned_tables: Dict of cleaned DataFrames
    - df_audit_trail: Complete log of every repair and quarantine action
    - metrics_summary: Dict of before/after counts, removed, repaired, and retained records
    """
    root = get_project_root()
    config = load_config()
    
    if input_dir is None:
        input_dir = root / "data" / "raw_with_quality_issues"
    if output_dir is None:
        output_dir = root / config["paths"]["processed_data_dir"]
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_dir} for automated cleaning...")
    df_products = pd.read_csv(input_dir / "products.csv")
    df_suppliers = pd.read_csv(input_dir / "suppliers.csv")
    df_warehouses = pd.read_csv(input_dir / "warehouses.csv")
    df_markets = pd.read_csv(input_dir / "markets.csv")
    df_date = pd.read_csv(input_dir / "date.csv")
    df_events = pd.read_csv(input_dir / "event_log.csv")
    df_demand = pd.read_csv(input_dir / "demand.csv")
    df_inventory = pd.read_csv(input_dir / "inventory.csv")
    df_pos = pd.read_csv(input_dir / "purchase_orders.csv")
    df_transport = pd.read_csv(input_dir / "transport.csv")
    
    audit_trail: List[Dict[str, Any]] = []
    
    before_metrics = {
        "demand_rows": len(df_demand),
        "inventory_rows": len(df_inventory),
        "po_rows": len(df_pos),
        "transport_rows": len(df_transport)
    }
    
    # Valid key reference sets
    valid_products = set(df_products["product_id"].dropna().unique())
    valid_suppliers = set(df_suppliers["supplier_id"].dropna().unique())
    valid_warehouses = set(df_warehouses["warehouse_id"].dropna().unique())
    valid_dates = set(df_date["date"].dropna().unique())
    cost_lookup = df_products.set_index("product_id")["unit_cost"].to_dict()
    
    # -------------------------------------------------------------
    # 1. Clean Customer Demand Table
    # -------------------------------------------------------------
    logger.info("Cleaning demand records...")
    # Step A: Deduplication
    dup_demand_count = int(df_demand.duplicated().sum())
    if dup_demand_count > 0:
        df_demand = df_demand.drop_duplicates(keep="first").reset_index(drop=True)
        audit_trail.append({
            "table_name": "demand.csv",
            "action": "DEDUPLICATE",
            "records_affected": dup_demand_count,
            "rule_id": "DQ-003",
            "reason": "Removed exact duplicate demand rows.",
            "disposition": "REMOVED"
        })
        
    # Step B: Remove Null Keys & Referential Orphans
    invalid_prod_mask = df_demand["product_id"].isna() | (~df_demand["product_id"].isin(valid_products))
    n_invalid_prod = int(invalid_prod_mask.sum())
    if n_invalid_prod > 0:
        df_demand = df_demand[~invalid_prod_mask].reset_index(drop=True)
        audit_trail.append({
            "table_name": "demand.csv",
            "action": "QUARANTINE_NULL_OR_ORPHAN_KEY",
            "records_affected": n_invalid_prod,
            "rule_id": "DQ-001/005",
            "reason": "Removed demand records with null or non-existent product IDs.",
            "disposition": "REMOVED"
        })
        
    # Step C: Remove Invalid Dates
    invalid_date_mask = df_demand["date"].isna() | (~df_demand["date"].isin(valid_dates))
    n_invalid_dates = int(invalid_date_mask.sum())
    if n_invalid_dates > 0:
        df_demand = df_demand[~invalid_date_mask].reset_index(drop=True)
        audit_trail.append({
            "table_name": "demand.csv",
            "action": "QUARANTINE_INVALID_DATE",
            "records_affected": n_invalid_dates,
            "rule_id": "DQ-002",
            "reason": "Removed demand records with malformed date timestamps.",
            "disposition": "REMOVED"
        })
        
    # Step D: Repair Negative Demand
    neg_demand_mask = df_demand["demand_requested"] < 0
    n_neg_demand = int(neg_demand_mask.sum())
    if n_neg_demand > 0:
        df_demand.loc[neg_demand_mask, "demand_requested"] = 0
        audit_trail.append({
            "table_name": "demand.csv",
            "action": "REPAIR_NEGATIVE_VALUE",
            "records_affected": n_neg_demand,
            "rule_id": "DQ-004",
            "reason": "Clamped negative demand requested values to zero floor.",
            "disposition": "REPAIRED"
        })
        
    # Step E: Detect Outliers & Flag (Retain without deletion)
    z_scores = (df_demand["demand_requested"] - df_demand["demand_requested"].mean()) / df_demand["demand_requested"].std()
    outlier_mask = z_scores > 5.0
    n_outliers = int(outlier_mask.sum())
    df_demand["outlier_flag"] = outlier_mask.astype(int)
    if n_outliers > 0:
        audit_trail.append({
            "table_name": "demand.csv",
            "action": "ANNOTATE_OUTLIER",
            "records_affected": n_outliers,
            "rule_id": "DQ-007",
            "reason": "Flagged statistical demand outliers for analytical awareness.",
            "disposition": "RETAINED_WITH_FLAG"
        })
        
    # Step F: Explicit non-event representation (avoid null ambiguous representation)
    df_demand["event_id"] = df_demand["event_id"].fillna("NONE").replace("", "NONE")

    # -------------------------------------------------------------
    # 2. Clean Purchase Orders Table
    # -------------------------------------------------------------
    logger.info("Cleaning purchase orders...")
    # Step A: Deduplicate
    dup_po_count = int(df_pos.duplicated().sum())
    if dup_po_count > 0:
        df_pos = df_pos.drop_duplicates(keep="first").reset_index(drop=True)
        audit_trail.append({
            "table_name": "purchase_orders.csv",
            "action": "DEDUPLICATE",
            "records_affected": dup_po_count,
            "rule_id": "DQ-003",
            "reason": "Removed exact duplicate purchase order rows.",
            "disposition": "REMOVED"
        })
        
    # Step B: Deduplicate Conflicting Primary Key (po_id)
    dup_poid_mask = df_pos.duplicated(subset=["po_id"], keep="first")
    n_dup_poid = int(dup_poid_mask.sum())
    if n_dup_poid > 0:
        df_pos = df_pos[~dup_poid_mask].reset_index(drop=True)
        audit_trail.append({
            "table_name": "purchase_orders.csv",
            "action": "DEDUPLICATE_PRIMARY_KEY",
            "records_affected": n_dup_poid,
            "rule_id": "DQ-003",
            "reason": "Quarantined conflicting duplicate po_id entries, preserving first authenticated record.",
            "disposition": "REMOVED"
        })
        
    # Step C: Remove Null / Orphan Supplier IDs
    invalid_sup_mask = df_pos["supplier_id"].isna() | (~df_pos["supplier_id"].isin(valid_suppliers))
    n_invalid_sup = int(invalid_sup_mask.sum())
    if n_invalid_sup > 0:
        df_pos = df_pos[~invalid_sup_mask].reset_index(drop=True)
        audit_trail.append({
            "table_name": "purchase_orders.csv",
            "action": "QUARANTINE_NULL_OR_ORPHAN_KEY",
            "records_affected": n_invalid_sup,
            "rule_id": "DQ-001/005",
            "reason": "Removed purchase orders referencing null or non-existent supplier IDs.",
            "disposition": "REMOVED"
        })
        
    # Step D: Repair Negative Unit Costs
    neg_cost_mask = df_pos["unit_cost"] <= 0
    n_neg_cost = int(neg_cost_mask.sum())
    if n_neg_cost > 0:
        for idx in df_pos.index[neg_cost_mask]:
            pid = df_pos.at[idx, "product_id"]
            restored_cost = cost_lookup.get(pid, 25.0)
            df_pos.at[idx, "unit_cost"] = restored_cost
            df_pos.at[idx, "total_procurement_cost"] = round(df_pos.at[idx, "quantity_received"] * restored_cost, 2)
        audit_trail.append({
            "table_name": "purchase_orders.csv",
            "action": "REPAIR_NEGATIVE_PRICE",
            "records_affected": n_neg_cost,
            "rule_id": "DQ-004",
            "reason": "Restored negative unit cost on purchase orders using product master standard cost.",
            "disposition": "REPAIRED"
        })

    # -------------------------------------------------------------
    # 3. Clean Inventory Ledger
    # -------------------------------------------------------------
    logger.info("Cleaning inventory ledger...")
    # Step A: Deduplication
    dup_inv_count = int(df_inventory.duplicated().sum())
    if dup_inv_count > 0:
        df_inventory = df_inventory.drop_duplicates(keep="first").reset_index(drop=True)
        audit_trail.append({
            "table_name": "inventory.csv",
            "action": "DEDUPLICATE",
            "records_affected": dup_inv_count,
            "rule_id": "DQ-003",
            "reason": "Removed duplicate inventory records.",
            "disposition": "REMOVED"
        })
        
    # Step B: Repair Impossible Inventory Values (Negative or Huge Overflow)
    # Strict physical balance equation: ending = max(0, beginning + received - fulfilled)
    inv_calc = (
        df_inventory["beginning_inventory"]
        + df_inventory["po_received"]
        - df_inventory["demand_fulfilled"]
    ).clip(lower=0)
    
    impossible_inv_mask = (df_inventory["ending_inventory"] < 0) | (df_inventory["ending_inventory"] != inv_calc)
    n_impossible_inv = int(impossible_inv_mask.sum())
    if n_impossible_inv > 0:
        df_inventory["ending_inventory"] = inv_calc
        # Recalculate holding cost
        daily_holding_rate = 0.20 / 365.25
        prod_cost_map = df_products.set_index("product_id")["unit_cost"].to_dict()
        unit_costs = df_inventory["product_id"].map(prod_cost_map)
        df_inventory["holding_cost"] = (df_inventory["ending_inventory"] * unit_costs * daily_holding_rate).round(4)
        
        audit_trail.append({
            "table_name": "inventory.csv",
            "action": "REPAIR_INVENTORY_BALANCE",
            "records_affected": n_impossible_inv,
            "rule_id": "DQ-006",
            "reason": "Re-derived ending inventory from beginning stock, receipts, and demand fulfillment.",
            "disposition": "REPAIRED"
        })

    # -------------------------------------------------------------
    # 4. Clean Transport Shipments
    # -------------------------------------------------------------
    # Filter transport records so they match remaining valid POs
    valid_poids = set(df_pos["po_id"].unique())
    orphan_trans_mask = ~df_transport["po_id"].isin(valid_poids)
    n_orphan_trans = int(orphan_trans_mask.sum())
    if n_orphan_trans > 0:
        df_transport = df_transport[~orphan_trans_mask].reset_index(drop=True)
        audit_trail.append({
            "table_name": "transport.csv",
            "action": "QUARANTINE_ORPHAN_TRANSPORT",
            "records_affected": n_orphan_trans,
            "rule_id": "DQ-005",
            "reason": "Removed transport shipments whose associated PO was quarantined.",
            "disposition": "REMOVED"
        })

    # Save cleaned tables to output_dir
    logger.info(f"Saving certified clean tables to {output_dir}...")
    cleaned_tables = {
        "products.csv": df_products,
        "suppliers.csv": df_suppliers,
        "warehouses.csv": df_warehouses,
        "markets.csv": df_markets,
        "date.csv": df_date,
        "event_log.csv": df_events,
        "demand.csv": df_demand,
        "inventory.csv": df_inventory,
        "purchase_orders.csv": df_pos,
        "transport.csv": df_transport
    }
    
    for filename, df in cleaned_tables.items():
        df.to_csv(output_dir / filename, index=False)
        
    df_audit = pd.DataFrame(audit_trail)
    df_audit.to_csv(output_dir / "audit_trail_log.csv", index=False)
    
    total_removed = sum(row["records_affected"] for row in audit_trail if row["disposition"] == "REMOVED")
    total_repaired = sum(row["records_affected"] for row in audit_trail if row["disposition"] == "REPAIRED")
    total_retained = sum([len(df) for df in cleaned_tables.values()])
    
    metrics_summary = {
        "before_rows": before_metrics,
        "after_rows": {
            "demand_rows": len(df_demand),
            "inventory_rows": len(df_inventory),
            "po_rows": len(df_pos),
            "transport_rows": len(df_transport)
        },
        "records_removed": int(total_removed),
        "records_repaired": int(total_repaired),
        "records_retained": int(total_retained),
        "audit_actions_count": len(audit_trail)
    }
    
    logger.info(f"Cleaning complete. Removed: {total_removed:,}, Repaired: {total_repaired:,}, Retained: {total_retained:,}.")
    return cleaned_tables, df_audit, metrics_summary
