"""
Test Suite for Phase 2: Synthetic Operations Data Generation & Physical Consistency.
Verifies entity counts, physical inventory balance equation, non-negativity,
reconciliation of demand/deliveries, supplier constraints, referential integrity, and determinism.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.utils.config import get_project_root, load_config
from src.data_generation.generator import run_data_generation


@pytest.fixture(scope="module")
def data_paths():
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    return {
        "products": raw_dir / "products.csv",
        "suppliers": raw_dir / "suppliers.csv",
        "warehouses": raw_dir / "warehouses.csv",
        "markets": raw_dir / "markets.csv",
        "date": raw_dir / "date.csv",
        "events": raw_dir / "event_log.csv",
        "demand": raw_dir / "demand.csv",
        "inventory": raw_dir / "inventory.csv",
        "pos": raw_dir / "purchase_orders.csv",
        "transport": raw_dir / "transport.csv",
        "manifest": raw_dir / "generation_manifest.json"
    }


def test_manifest_and_files_exist(data_paths):
    """Verify all raw CSVs and manifest JSON exist on disk."""
    for name, p in data_paths.items():
        assert p.exists(), f"File {name} not found at {p}"


def test_entity_counts(data_paths):
    """Verify exact entity dimensions required by specification."""
    df_products = pd.read_csv(data_paths["products"])
    df_suppliers = pd.read_csv(data_paths["suppliers"])
    df_warehouses = pd.read_csv(data_paths["warehouses"])
    df_markets = pd.read_csv(data_paths["markets"])
    df_date = pd.read_csv(data_paths["date"])
    df_events = pd.read_csv(data_paths["events"])

    assert len(df_products) == 60, "Must have exactly 60 products"
    assert df_products["category"].nunique() == 6, "Must have exactly 6 categories"
    assert len(df_suppliers) == 8, "Must have exactly 8 suppliers"
    assert len(df_warehouses) == 4, "Must have exactly 4 warehouses"
    assert len(df_markets) == 6, "Must have exactly 6 markets"
    assert len(df_date) == 731, "Must span exactly 731 calendar days (2024 leap year + 2025)"
    assert len(df_events) >= 5, "Must have at least 5 macro events"


def test_referential_integrity(data_paths):
    """Verify all transactional foreign keys map to existing primary keys."""
    df_products = pd.read_csv(data_paths["products"])
    df_suppliers = pd.read_csv(data_paths["suppliers"])
    df_warehouses = pd.read_csv(data_paths["warehouses"])
    df_markets = pd.read_csv(data_paths["markets"])
    df_demand = pd.read_csv(data_paths["demand"])
    df_inventory = pd.read_csv(data_paths["inventory"])
    df_pos = pd.read_csv(data_paths["pos"])

    valid_prod_ids = set(df_products["product_id"])
    valid_sup_ids = set(df_suppliers["supplier_id"])
    valid_wh_ids = set(df_warehouses["warehouse_id"])
    valid_mkt_ids = set(df_markets["market_id"])

    # Demand table keys
    assert set(df_demand["product_id"]).issubset(valid_prod_ids)
    assert set(df_demand["warehouse_id"]).issubset(valid_wh_ids)
    assert set(df_demand["market_id"]).issubset(valid_mkt_ids)

    # Inventory table keys
    assert set(df_inventory["product_id"]).issubset(valid_prod_ids)
    assert set(df_inventory["warehouse_id"]).issubset(valid_wh_ids)

    # Purchase orders keys
    assert set(df_pos["product_id"]).issubset(valid_prod_ids)
    assert set(df_pos["supplier_id"]).issubset(valid_sup_ids)
    assert set(df_pos["warehouse_id"]).issubset(valid_wh_ids)


def test_non_negative_demand(data_paths):
    """Demand requested must be strictly non-negative."""
    df_demand = pd.read_csv(data_paths["demand"])
    assert (df_demand["demand_requested"] >= 0).all(), "Demand requested contains negative values"
    assert df_demand["demand_requested"].sum() > 0, "Total demand cannot be 0"


def test_physical_inventory_balance_reconciliation(data_paths):
    """
    Core Physical Ledger Equation:
    Beginning Inventory + PO Received - Demand Fulfilled - Ending Inventory == 0
    Must hold strictly for 100% of records.
    """
    df_inventory = pd.read_csv(data_paths["inventory"])
    diff = (
        df_inventory["beginning_inventory"]
        + df_inventory["po_received"]
        - df_inventory["demand_fulfilled"]
        - df_inventory["ending_inventory"]
    ).abs()
    assert (diff == 0).all(), f"Inventory balance equation failed on {np.sum(diff != 0)} records"


def test_non_negative_inventory(data_paths):
    """Ending inventory must be strictly non-negative."""
    df_inventory = pd.read_csv(data_paths["inventory"])
    assert (df_inventory["ending_inventory"] >= 0).all(), "Ending inventory contains negative values"


def test_demand_fulfillment_bounds(data_paths):
    """Demand fulfilled cannot exceed available stock."""
    df_inventory = pd.read_csv(data_paths["inventory"])
    available_stock = df_inventory["beginning_inventory"] + df_inventory["po_received"]
    assert (df_inventory["demand_fulfilled"] <= available_stock).all(), (
        "Demand fulfilled exceeds available physical stock"
    )


def test_stockout_logic(data_paths):
    """
    Stockout flag must be 1 if and only if there was an actual shortage.
    When stockout_flag == 1, stockout_quantity must be > 0.
    """
    df_inventory = pd.read_csv(data_paths["inventory"])
    stockout_rows = df_inventory[df_inventory["stockout_flag"] == 1]
    non_stockout_rows = df_inventory[df_inventory["stockout_flag"] == 0]

    assert (stockout_rows["stockout_quantity"] > 0).all(), (
        "Stockout flag is 1 but stockout quantity is not positive"
    )
    assert (non_stockout_rows["stockout_quantity"] == 0).all(), (
        "Stockout flag is 0 but stockout quantity is positive"
    )


def test_purchase_order_consistency(data_paths):
    """
    Purchase Orders must have:
    - actual delivery date >= order date
    - quantity received <= quantity ordered
    - lead times > 0
    - positive unit cost
    """
    df_pos = pd.read_csv(data_paths["pos"])
    assert (df_pos["actual_delivery_date"] >= df_pos["order_date"]).all(), (
        "Actual delivery date is earlier than order date"
    )
    assert (df_pos["quantity_received"] <= df_pos["quantity_ordered"]).all(), (
        "Received quantity exceeds ordered quantity"
    )
    assert (df_pos["actual_lead_time_days"] >= 1).all(), "Actual lead time must be >= 1 day"
    assert (df_pos["unit_cost"] > 0).all(), "Unit cost must be positive"


def test_event_log_validity(data_paths):
    """Event start dates must be <= end dates and lie within the calendar horizon."""
    df_events = pd.read_csv(data_paths["events"])
    assert (df_events["start_date"] <= df_events["end_date"]).all(), "Event start date after end date"
    assert (df_events["demand_multiplier"] > 0).all(), "Demand multiplier must be positive"
    assert (df_events["lead_time_multiplier"] >= 1.0).all(), "Lead time multiplier must be >= 1.0"


def test_cost_calculations(data_paths):
    """Financial columns must be non-negative and properly computed."""
    df_inventory = pd.read_csv(data_paths["inventory"])
    df_pos = pd.read_csv(data_paths["pos"])

    assert (df_inventory["holding_cost"] >= 0).all(), "Negative holding cost detected"
    assert (df_inventory["stockout_cost"] >= 0).all(), "Negative stockout cost detected"
    assert (df_pos["total_procurement_cost"] > 0).all(), "Non-positive procurement cost detected"
    assert (df_pos["transport_cost"] >= 0).all(), "Negative transport cost detected"


def test_cost_reconciliation_and_formulas(data_paths):
    """
    Dedicated verification of exact cost formulas and total cost reconciliation:
    1. holding_cost = round(ending_inventory * unit_cost * (0.20 / 365.25), 4)
    2. stockout_cost = round(lost_sales_quantity * selling_price * 1.50, 4)
    3. procurement_cost = round(quantity_received * unit_cost, 2)
    4. total_cost = procurement_cost + holding_cost + stockout_cost + transport_cost (diff == 0)
    """
    df_inventory = pd.read_csv(data_paths["inventory"])
    df_pos = pd.read_csv(data_paths["pos"])
    df_products = pd.read_csv(data_paths["products"])

    # Merge product financial attributes for formula verification
    df_inv_merged = df_inventory.merge(
        df_products[["product_id", "unit_cost", "selling_price"]],
        on="product_id",
        how="left"
    )

    # 1. Holding cost formula
    expected_holding = (
        df_inv_merged["ending_inventory"] * df_inv_merged["unit_cost"] * (0.20 / 365.25)
    ).round(4)
    holding_diff = (df_inv_merged["holding_cost"] - expected_holding).abs().max()
    assert holding_diff == 0.0, f"Holding cost formula mismatch: max diff {holding_diff}"

    # 2. Stockout penalty formula
    expected_stockout = (
        df_inv_merged["lost_sales_quantity"] * df_inv_merged["selling_price"] * 1.50
    ).round(4)
    stockout_diff = (df_inv_merged["stockout_cost"] - expected_stockout).abs().max()
    assert stockout_diff == 0.0, f"Stockout penalty cost formula mismatch: max diff {stockout_diff}"

    # 3. Procurement cost formula
    expected_proc = (df_pos["quantity_received"] * df_pos["unit_cost"]).round(2)
    proc_diff = (df_pos["total_procurement_cost"] - expected_proc).abs().max()
    assert proc_diff == 0.0, f"Procurement cost formula mismatch: max diff {proc_diff}"

    # 4. Total operational cost summation
    p = df_pos["total_procurement_cost"].sum()
    h = df_inventory["holding_cost"].sum()
    s = df_inventory["stockout_cost"].sum()
    t = df_pos["transport_cost"].sum()
    total_calc = round(p + h + s + t, 2)

    with open(data_paths["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_total = manifest["operational_metrics"]["costs"]["total_operational_cost"]

    assert abs(total_calc - manifest_total) <= 0.01, (
        f"Total operational cost does not reconcile: calculated {total_calc} vs manifest {manifest_total}"
    )


def test_reproducibility(tmp_path):
    """Verify that running generation with scale=0.2 and same seed produces identical row sums in isolated directory."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    m1 = run_data_generation(seed=999, scale=0.2, output_dir=dir1)
    m2 = run_data_generation(seed=999, scale=0.2, output_dir=dir2)

    assert m1["operational_metrics"]["total_demand_requested_units"] == (
        m2["operational_metrics"]["total_demand_requested_units"]
    ), "Data generation is not deterministic across identical random seeds"
    assert m1["operational_metrics"]["total_demand_fulfilled_units"] == (
        m2["operational_metrics"]["total_demand_fulfilled_units"]
    ), "Inventory simulation is not deterministic across identical random seeds"
