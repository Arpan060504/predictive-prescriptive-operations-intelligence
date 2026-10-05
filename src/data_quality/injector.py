"""
Controlled Quality Issue Injection Engine for PPOI.
Takes clean synthetic data from data/raw/ and introduces deterministic, realistic data quality flaws
into a dedicated data/raw_with_quality_issues/ directory.
Logs every injected flaw to injected_issues_log.csv for complete auditability.
Preserves the pristine data/raw/ dataset completely untouched.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("QualityInjector")


def run_issue_injection(seed: int = 42) -> pd.DataFrame:
    """
    Executes controlled quality flaw injection and returns the comprehensive audit log DataFrame.
    """
    rng = np.random.default_rng(seed)
    root = get_project_root()
    config = load_config()
    
    raw_dir = root / config["paths"]["raw_data_dir"]
    issues_dir = root / "data" / "raw_with_quality_issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading clean operational tables from data/raw/...")
    df_products = pd.read_csv(raw_dir / "products.csv")
    df_suppliers = pd.read_csv(raw_dir / "suppliers.csv")
    df_warehouses = pd.read_csv(raw_dir / "warehouses.csv")
    df_markets = pd.read_csv(raw_dir / "markets.csv")
    df_date = pd.read_csv(raw_dir / "date.csv")
    df_events = pd.read_csv(raw_dir / "event_log.csv")
    df_demand = pd.read_csv(raw_dir / "demand.csv")
    df_inventory = pd.read_csv(raw_dir / "inventory.csv")
    df_pos = pd.read_csv(raw_dir / "purchase_orders.csv")
    df_transport = pd.read_csv(raw_dir / "transport.csv")
    
    injected_log: List[Dict[str, str]] = []
    issue_idx = 1
    
    # ---------------------------------------------------------
    # 1. Null Primary / Foreign Keys (Demand & POs)
    # ---------------------------------------------------------
    logger.info("Injecting null key values...")
    null_demand_indices = rng.choice(df_demand.index, size=60, replace=False)
    for idx in null_demand_indices:
        orig = df_demand.at[idx, "product_id"]
        df_demand.at[idx, "product_id"] = np.nan
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(idx),
            "column_name": "product_id",
            "original_value": str(orig),
            "corrupted_value": "NULL",
            "issue_category": "NULL_KEY",
            "severity": "ERROR",
            "description": "Product ID set to NaN in customer demand request."
        })
        issue_idx += 1
        
    null_po_indices = rng.choice(df_pos.index, size=35, replace=False)
    for idx in null_po_indices:
        orig = df_pos.at[idx, "supplier_id"]
        df_pos.at[idx, "supplier_id"] = np.nan
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "purchase_orders.csv",
            "row_index": str(idx),
            "column_name": "supplier_id",
            "original_value": str(orig),
            "corrupted_value": "NULL",
            "issue_category": "NULL_KEY",
            "severity": "ERROR",
            "description": "Supplier ID set to NaN on replenishment purchase order."
        })
        issue_idx += 1

    # ---------------------------------------------------------
    # 2. Invalid Dates / Malformed Timestamps
    # ---------------------------------------------------------
    logger.info("Injecting invalid dates and malformed strings...")
    invalid_date_indices = rng.choice(df_demand.index, size=45, replace=False)
    for idx in invalid_date_indices:
        orig = df_demand.at[idx, "date"]
        corrupted = "2024-02-30" if (issue_idx % 2 == 0) else "INVALID_DATE"
        df_demand.at[idx, "date"] = corrupted
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(idx),
            "column_name": "date",
            "original_value": str(orig),
            "corrupted_value": corrupted,
            "issue_category": "INVALID_DATE",
            "severity": "ERROR",
            "description": f"Date timestamp corrupted to invalid string '{corrupted}'."
        })
        issue_idx += 1

    # ---------------------------------------------------------
    # 3. Exact Duplicate Records
    # ---------------------------------------------------------
    logger.info("Injecting exact duplicate records...")
    dup_demand_samples = df_demand.sample(n=120, random_state=seed)
    for orig_idx in dup_demand_samples.index:
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(orig_idx),
            "column_name": "ALL",
            "original_value": "ORIGINAL_ROW",
            "corrupted_value": "DUPLICATE_ROW",
            "issue_category": "DUPLICATE_RECORD",
            "severity": "ERROR",
            "description": "Appended exact duplicate demand transaction row."
        })
        issue_idx += 1
    df_demand = pd.concat([df_demand, dup_demand_samples], ignore_index=True)

    dup_po_samples = df_pos.sample(n=50, random_state=seed)
    for orig_idx in dup_po_samples.index:
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "purchase_orders.csv",
            "row_index": str(orig_idx),
            "column_name": "ALL",
            "original_value": "ORIGINAL_ROW",
            "corrupted_value": "DUPLICATE_ROW",
            "issue_category": "DUPLICATE_RECORD",
            "severity": "ERROR",
            "description": "Appended exact duplicate purchase order record."
        })
        issue_idx += 1
    df_pos = pd.concat([df_pos, dup_po_samples], ignore_index=True)

    # ---------------------------------------------------------
    # 4. Duplicate Primary Keys with Conflicting Quantities
    # ---------------------------------------------------------
    logger.info("Injecting duplicate transaction IDs with conflicting values...")
    conflict_pos = df_pos.head(25).copy()
    conflict_pos["quantity_ordered"] = conflict_pos["quantity_ordered"] + 500
    for idx, row in conflict_pos.iterrows():
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "purchase_orders.csv",
            "row_index": str(idx),
            "column_name": "po_id",
            "original_value": str(row["po_id"]),
            "corrupted_value": f"{row['po_id']}_CONFLICT_QTY",
            "issue_category": "DUPLICATE_TRANSACTION_KEY",
            "severity": "ERROR",
            "description": f"Injected duplicate po_id '{row['po_id']}' with conflicting quantity (+500 units)."
        })
        issue_idx += 1
    df_pos = pd.concat([df_pos, conflict_pos], ignore_index=True)

    # ---------------------------------------------------------
    # 5. Negative Quantities and Negative Prices
    # ---------------------------------------------------------
    logger.info("Injecting negative demand and negative prices...")
    neg_demand_indices = rng.choice(df_demand.index[:10000], size=30, replace=False)
    for idx in neg_demand_indices:
        orig = df_demand.at[idx, "demand_requested"]
        df_demand.at[idx, "demand_requested"] = -int(orig) if orig > 0 else -18
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(idx),
            "column_name": "demand_requested",
            "original_value": str(orig),
            "corrupted_value": str(df_demand.at[idx, "demand_requested"]),
            "issue_category": "NEGATIVE_NUMERIC",
            "severity": "ERROR",
            "description": "Negative demand requested value created."
        })
        issue_idx += 1

    neg_price_indices = rng.choice(df_pos.index[:2000], size=15, replace=False)
    for idx in neg_price_indices:
        orig = df_pos.at[idx, "unit_cost"]
        df_pos.at[idx, "unit_cost"] = -round(float(orig), 2)
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "purchase_orders.csv",
            "row_index": str(idx),
            "column_name": "unit_cost",
            "original_value": str(orig),
            "corrupted_value": str(df_pos.at[idx, "unit_cost"]),
            "issue_category": "NEGATIVE_NUMERIC",
            "severity": "ERROR",
            "description": "Negative unit cost created on purchase order."
        })
        issue_idx += 1

    # ---------------------------------------------------------
    # 6. Impossible Inventory Values (Negative & Extreme Overflow)
    # ---------------------------------------------------------
    logger.info("Injecting impossible inventory values...")
    inv_err_indices = rng.choice(df_inventory.index, size=40, replace=False)
    for i, idx in enumerate(inv_err_indices):
        orig = df_inventory.at[idx, "ending_inventory"]
        corrupted = -85 if i < 20 else 1250000 # Negative or 1.25M units exceeding warehouse cap
        df_inventory.at[idx, "ending_inventory"] = corrupted
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "inventory.csv",
            "row_index": str(idx),
            "column_name": "ending_inventory",
            "original_value": str(orig),
            "corrupted_value": str(corrupted),
            "issue_category": "IMPOSSIBLE_INVENTORY",
            "severity": "ERROR",
            "description": f"Ending inventory corrupted to physically impossible value: {corrupted}."
        })
        issue_idx += 1

    # ---------------------------------------------------------
    # 7. Referential Integrity Violations (Orphan IDs)
    # ---------------------------------------------------------
    logger.info("Injecting referential integrity violations...")
    orphan_demand_indices = rng.choice(df_demand.index, size=50, replace=False)
    for idx in orphan_demand_indices:
        orig = df_demand.at[idx, "product_id"]
        df_demand.at[idx, "product_id"] = "SKU-999"
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(idx),
            "column_name": "product_id",
            "original_value": str(orig),
            "corrupted_value": "SKU-999",
            "issue_category": "REFERENTIAL_VIOLATION",
            "severity": "ERROR",
            "description": "Product ID set to non-existent 'SKU-999' not present in dim_product."
        })
        issue_idx += 1

    orphan_po_indices = rng.choice(df_pos.index, size=30, replace=False)
    for idx in orphan_po_indices:
        orig = df_pos.at[idx, "supplier_id"]
        df_pos.at[idx, "supplier_id"] = "SUP-99"
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "purchase_orders.csv",
            "row_index": str(idx),
            "column_name": "supplier_id",
            "original_value": str(orig),
            "corrupted_value": "SUP-99",
            "issue_category": "REFERENTIAL_VIOLATION",
            "severity": "ERROR",
            "description": "Supplier ID set to non-existent 'SUP-99' not present in dim_supplier."
        })
        issue_idx += 1

    # ---------------------------------------------------------
    # 8. Temporal Gaps (Dropping date ranges for a SKU-Warehouse)
    # ---------------------------------------------------------
    logger.info("Injecting temporal gaps...")
    target_pid = "SKU-005"
    target_wid = "WH-01"
    gap_dates = pd.date_range("2024-05-01", "2024-05-14").strftime("%Y-%m-%d")
    drop_mask = (
        (df_demand["product_id"] == target_pid) &
        (df_demand["warehouse_id"] == target_wid) &
        (df_demand["date"].isin(gap_dates))
    )
    n_dropped = int(drop_mask.sum())
    df_demand = df_demand[~drop_mask].reset_index(drop=True)
    injected_log.append({
        "issue_id": f"INJ-{issue_idx:05d}",
        "table_name": "demand.csv",
        "row_index": f"{target_pid}_{target_wid}_2024-05-01_to_14",
        "column_name": "date",
        "original_value": f"{n_dropped} dates present",
        "corrupted_value": "DELETED_RECORDS",
        "issue_category": "TEMPORAL_GAP",
        "severity": "WARNING",
        "description": f"Dropped {n_dropped} consecutive daily observations creating a 14-day temporal gap for {target_pid} at {target_wid}."
    })
    issue_idx += 1

    # ---------------------------------------------------------
    # 9. Extreme Statistical Outliers
    # ---------------------------------------------------------
    logger.info("Injecting extreme statistical outliers...")
    outlier_indices = rng.choice(df_demand.index[:5000], size=10, replace=False)
    for idx in outlier_indices:
        orig = df_demand.at[idx, "demand_requested"]
        extreme_val = 3200 # Normal range is 10 - 150
        df_demand.at[idx, "demand_requested"] = extreme_val
        injected_log.append({
            "issue_id": f"INJ-{issue_idx:05d}",
            "table_name": "demand.csv",
            "row_index": str(idx),
            "column_name": "demand_requested",
            "original_value": str(orig),
            "corrupted_value": str(extreme_val),
            "issue_category": "STATISTICAL_OUTLIER",
            "severity": "WARNING",
            "description": f"Extreme demand spike injected ({extreme_val} units vs baseline {orig})."
        })
        issue_idx += 1

    # Save all modified tables to data/raw_with_quality_issues/
    logger.info("Saving corrupted operational tables to data/raw_with_quality_issues/...")
    tables = {
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
    
    for filename, df in tables.items():
        df.to_csv(issues_dir / filename, index=False)
        
    df_log = pd.DataFrame(injected_log)
    df_log.to_csv(issues_dir / "injected_issues_log.csv", index=False)
    
    logger.info(f"Injection complete. Total quality issues logged: {len(df_log)}.")
    return df_log


if __name__ == "__main__":
    run_issue_injection()
