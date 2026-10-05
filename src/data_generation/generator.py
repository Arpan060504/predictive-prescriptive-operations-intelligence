"""
Master Data Generation Orchestrator for PPOI.
Executes end-to-end generation of master entities, daily demand, sequential inventory ledger,
purchase orders, transport shipments, and macro disruption event logs.
Emits data/raw CSVs, generation manifest JSON, and reports/data_generation_report.md.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.data_generation.entities import (
    generate_date_dimension,
    generate_event_log,
    generate_markets,
    generate_products,
    generate_suppliers,
    generate_warehouses,
)
from src.data_generation.demand import generate_demand
from src.data_generation.inventory import simulate_inventory_and_orders
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("DataGenerator")


def run_data_generation(
    seed: Optional[int] = None,
    scale: float = 1.0,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Runs the complete reproducible synthetic data generation pipeline.
    """
    start_time = time.time()
    config = load_config()
    
    if seed is None:
        seed = int(config.get("project", {}).get("random_seed", 42))
        
    logger.info(f"Initiating synthetic data generation with seed={seed}, scale={scale}...")
    
    root_dir = get_project_root()
    if output_dir is not None:
        raw_dir = Path(output_dir)
        demo_dir = Path(output_dir) / "demo"
        reports_dir = Path(output_dir) / "reports"
    else:
        raw_dir = root_dir / config["paths"]["raw_data_dir"]
        demo_dir = root_dir / config["paths"]["demo_data_dir"]
        reports_dir = root_dir / config["paths"]["reports_dir"]
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Master Dimensions
    start_date = config["data_generation"]["start_date"]
    end_date = config["data_generation"]["end_date"]
    
    logger.info("Generating date calendar and macro event log...")
    df_date = generate_date_dimension(start_date=start_date, end_date=end_date)
    df_events = generate_event_log()
    
    logger.info("Generating warehouses, markets, suppliers, and products...")
    df_warehouses = generate_warehouses()
    df_markets = generate_markets()
    df_suppliers = generate_suppliers(seed=seed)
    df_products = generate_products(seed=seed)
    
    # Scale adjustment if user requested scale < 1.0 (e.g. for rapid dev testing)
    if scale < 1.0:
        n_prods = max(12, int(np.round(len(df_products) * scale)))
        selected_pids = df_products["product_id"].head(n_prods).tolist()
        df_products = df_products[df_products["product_id"].isin(selected_pids)].reset_index(drop=True)
        logger.info(f"Scaled product master down to {len(df_products)} SKUs for scale={scale}")
        
    # 2. Demand Requests
    logger.info("Generating customer demand requests across products, markets, and dates...")
    df_demand = generate_demand(
        df_dates=df_date,
        df_products=df_products,
        df_markets=df_markets,
        df_events=df_events,
        seed=seed
    )
    
    # 3. Inventory Ledger & Purchase Orders
    logger.info("Executing sequential inventory ledger and purchase order simulation...")
    df_inventory, df_pos, df_transport = simulate_inventory_and_orders(
        df_demand=df_demand,
        df_products=df_products,
        df_suppliers=df_suppliers,
        df_warehouses=df_warehouses,
        df_dates=df_date,
        df_events=df_events,
        seed=seed
    )
    
    # 4. Save to CSVs
    logger.info("Saving generated tables to data/raw/...")
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
    
    row_counts = {}
    for filename, df in tables.items():
        file_path = raw_dir / filename
        df.to_csv(file_path, index=False)
        row_counts[filename] = len(df)
        
        # Also copy a verified baseline to demo folder for fast zero-training dashboard launch
        demo_path = demo_dir / filename
        df.to_csv(demo_path, index=False)
        
    runtime_seconds = round(time.time() - start_time, 2)
    logger.info(f"All tables written to disk in {runtime_seconds}s.")
    
    # 5. Calculate Comprehensive Reconciliation & Operational Statistics
    total_demand_req = int(df_demand["demand_requested"].sum())
    total_demand_ful = int(df_inventory["demand_fulfilled"].sum())
    total_lost_sales = int(df_inventory["lost_sales_quantity"].sum())
    stockout_events = int((df_inventory["stockout_flag"] == 1).sum())
    total_inv_records = len(df_inventory)
    stockout_rate_pct = round((stockout_events / total_inv_records) * 100, 2)
    service_level_pct = round((total_demand_ful / max(1, total_demand_req)) * 100, 2)
    
    avg_inventory_units = round(float(df_inventory["ending_inventory"].mean()), 1)
    avg_supplier_delay_days = round(float(df_pos["delay_days"].mean()), 2)
    on_time_delivery_pct = round(float((df_pos["on_time_flag"] == 1).mean()) * 100, 2)
    otif_pct = round(float((df_pos["otif_flag"] == 1).mean()) * 100, 2)
    
    total_procurement_cost = round(float(df_pos["total_procurement_cost"].sum()), 2)
    total_holding_cost = round(float(df_inventory["holding_cost"].sum()), 2)
    total_stockout_cost = round(float(df_inventory["stockout_cost"].sum()), 2)
    total_transport_cost = round(float(df_pos["transport_cost"].sum()), 2)
    total_operational_cost = round(
        total_procurement_cost + total_holding_cost + total_stockout_cost + total_transport_cost, 2
    )
    
    # 6. Physical Reconciliation Check
    # Beginning + Received - Fulfilled == Ending Inventory
    reconciliation_diff = (
        df_inventory["beginning_inventory"]
        + df_inventory["po_received"]
        - df_inventory["demand_fulfilled"]
        - df_inventory["ending_inventory"]
    ).abs().max()
    
    reconciliation_passed = bool(reconciliation_diff == 0)
    
    total_operational_records = sum([
        len(df_demand), len(df_inventory), len(df_pos), len(df_transport)
    ])
    
    # 7. Write Manifest
    manifest = {
        "generation_timestamp": datetime.now().isoformat(),
        "random_seed": seed,
        "scale": scale,
        "generation_runtime_seconds": runtime_seconds,
        "date_range": {
            "start": start_date,
            "end": end_date,
            "total_days": len(df_date)
        },
        "entity_counts": {
            "products": len(df_products),
            "categories": int(df_products["category"].nunique()),
            "suppliers": len(df_suppliers),
            "warehouses": len(df_warehouses),
            "markets": len(df_markets),
            "macro_events": len(df_events)
        },
        "row_counts": row_counts,
        "total_operational_records": total_operational_records,
        "operational_metrics": {
            "total_demand_requested_units": total_demand_req,
            "total_demand_fulfilled_units": total_demand_ful,
            "total_lost_sales_units": total_lost_sales,
            "stockout_frequency_pct": stockout_rate_pct,
            "overall_service_level_pct": service_level_pct,
            "average_daily_inventory_units_per_sku_wh": avg_inventory_units,
            "average_supplier_delay_days": avg_supplier_delay_days,
            "supplier_on_time_rate_pct": on_time_delivery_pct,
            "supplier_otif_rate_pct": otif_pct,
            "costs": {
                "procurement_cost": total_procurement_cost,
                "holding_cost": total_holding_cost,
                "stockout_penalty_cost": total_stockout_cost,
                "transport_cost": total_transport_cost,
                "total_operational_cost": total_operational_cost
            }
        },
        "reconciliation_passed": reconciliation_passed
    }
    
    manifest_path = raw_dir / "generation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # Copy to demo manifest
    with open(demo_dir / "generation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 8. Generate reports/data_generation_report.md
    report_md = f"""# Data Generation Report: Synthetic Operations Environment

**Generated At:** {manifest['generation_timestamp']}  
**Random Seed:** `{seed}` | **Scale:** `{scale}` | **Runtime:** `{runtime_seconds}s`  
**Disclaimer:** *All operational data in this project is synthetic and generated for analytical demonstration.*

---

## 1. Dataset Scale & Record Counts

| Table | File Path | Record Count | Description |
| :--- | :--- | :--- | :--- |
| **Products Master** | `data/raw/products.csv` | {row_counts['products.csv']:,} | 60 SKUs across 6 distinct categories with cost, lead time, and storage constraints |
| **Suppliers Master** | `data/raw/suppliers.csv` | {row_counts['suppliers.csv']:,} | 8 suppliers with distinct reliability, cost, and capacity profiles |
| **Warehouses Master** | `data/raw/warehouses.csv` | {row_counts['warehouses.csv']:,} | 4 regional distribution centers with capacity limits and handling fees |
| **Markets Master** | `data/raw/markets.csv` | {row_counts['markets.csv']:,} | 6 demand regions with regional multipliers and annual growth rates |
| **Date Calendar** | `data/raw/date.csv` | {row_counts['date.csv']:,} | Daily calendar from {start_date} to {end_date} (731 days) with holiday & promo flags |
| **Macro Event Log** | `data/raw/event_log.csv` | {row_counts['event_log.csv']:,} | 6 deterministic disruptions and holiday demand surges |
| **Customer Demand** | `data/raw/demand.csv` | {row_counts['demand.csv']:,} | Daily SKU-Market demand requested |
| **Inventory Ledger** | `data/raw/inventory.csv` | {row_counts['inventory.csv']:,} | Daily SKU-Warehouse balance tracking beginning, received, fulfilled, and ending inventory |
| **Purchase Orders** | `data/raw/purchase_orders.csv` | {row_counts['purchase_orders.csv']:,} | Orders triggered under continuous review policy with supplier lead-time outcomes |
| **Transport Shipments** | `data/raw/transport.csv` | {row_counts['transport.csv']:,} | Freight transit movements between suppliers and warehouses |
| **TOTAL OPERATIONAL**| | **{total_operational_records:,}** | Operational and transactional records |

---

## 2. Category Distribution

| Category | Products Count | Base Daily Demand Range | Avg Unit Cost | Avg Unit Selling Price |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cat, grp in df_products.groupby("category"):
        report_md += f"| {cat} | {len(grp)} | {grp['base_demand'].min()} - {grp['base_demand'].max()} | ${grp['unit_cost'].mean():.2f} | ${grp['selling_price'].mean():.2f} |\n"
        
    report_md += f"""
---

## 3. Operational & Financial Performance Summary

- **Total Demand Requested:** {total_demand_req:,} units
- **Total Demand Fulfilled:** {total_demand_ful:,} units
- **Overall Service Level:** {service_level_pct}%
- **Stockout Frequency Rate:** {stockout_rate_pct}% of SKU-Warehouse-Days
- **Total Lost Sales:** {total_lost_sales:,} units
- **Average Daily Inventory (per SKU/WH):** {avg_inventory_units} units
- **Supplier On-Time Delivery Rate:** {on_time_delivery_pct}%
- **Supplier OTIF (On-Time In-Full) Rate:** {otif_pct}%
- **Average Supplier Lead Time Delay:** {avg_supplier_delay_days} days

### Financial Cost Breakdown
- **Procurement Cost:** ${total_procurement_cost:,.2f}
- **Inventory Holding Cost:** ${total_holding_cost:,.2f} (20% annualized)
- **Stockout Penalty Cost:** ${total_stockout_cost:,.2f} (1.5x unit price)
- **Transportation Cost:** ${total_transport_cost:,.2f}
- **Total Operational Cost:** ${total_operational_cost:,.2f}

---

## 4. Physical Reconciliation Verification

- **Equation Tested:** `Beginning Inventory + PO Received - Demand Fulfilled - Ending Inventory = 0`
- **Max Absolute Deviation:** `{reconciliation_diff}`
- **Physical Consistency Status:** `{'PASSED' if reconciliation_passed else 'FAILED'}`
- **Non-negative Inventory Guarantee:** `{'PASSED' if (df_inventory['ending_inventory'] >= 0).all() else 'FAILED'}`
- **Non-negative Demand Guarantee:** `{'PASSED' if (df_demand['demand_requested'] >= 0).all() else 'FAILED'}`
"""
    
    with open(reports_dir / "data_generation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    # Print formatted sanity check report
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Products: {len(df_products)}")
    print(f"Categories: {df_products['category'].nunique()}")
    print(f"Suppliers: {len(df_suppliers)}")
    print(f"Warehouses: {len(df_warehouses)}")
    print(f"Markets: {len(df_markets)}")
    print(f"Days: {len(df_date)} ({start_date} to {end_date})")
    print("-" * 60)
    print(f"Demand records: {row_counts['demand.csv']:,}")
    print(f"Inventory records: {row_counts['inventory.csv']:,}")
    print(f"PO records: {row_counts['purchase_orders.csv']:,}")
    print(f"Transport records: {row_counts['transport.csv']:,}")
    print(f"Total operational records: {total_operational_records:,}")
    print("-" * 60)
    print(f"Stockout rate: {stockout_rate_pct}%")
    print(f"Overall service level: {service_level_pct}%")
    print(f"Average inventory: {avg_inventory_units:,.1f} units")
    print(f"Average supplier delay: {avg_supplier_delay_days:.2f} days")
    print(f"Supplier OTIF rate: {otif_pct}%")
    print(f"Physical balance reconciliation: {'PASS' if reconciliation_passed else 'FAIL'}")
    print(f"Runtime: {runtime_seconds} seconds")
    print("=" * 60 + "\n")
    
    return manifest


if __name__ == "__main__":
    run_data_generation()
