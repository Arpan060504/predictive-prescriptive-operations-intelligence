"""
Database Builder Engine for PPOI.
Constructs the Star Schema SQLite database (database/operations.db) strictly from the certified
analytical dataset in data/processed/.
Enforces foreign key referential integrity, WAL mode, and high-performance B-tree indexes.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.database.schema import DDL_STATEMENTS, INDEX_STATEMENTS
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("DatabaseBuilder")


def build_database(
    processed_dir: Optional[Path] = None,
    db_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Builds database/operations.db from data/processed/ tables.
    Returns summary dict of execution runtime, row counts, and foreign key verification status.
    """
    start_time = time.time()
    root = get_project_root()
    config = load_config()

    if processed_dir is None:
        processed_dir = root / config["paths"]["processed_data_dir"]
    if db_file is None:
        db_file = root / config["paths"]["database_file"]

    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database file if present to guarantee clean build
    if db_file.exists():
        logger.info(f"Removing existing database at {db_file} for clean rebuild...")
        try:
            db_file.unlink()
        except Exception as e:
            logger.warning(f"Could not unlink {db_file} directly: {e}")

    logger.info(f"Connecting to SQLite database: {db_file}...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Optimization pragmas
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    # 1. Execute DDL
    logger.info("Executing Star Schema DDL statements...")
    for ddl in DDL_STATEMENTS:
        cursor.execute(ddl)
    conn.commit()

    # 2. Load Dimensions
    logger.info("Loading Dimension tables from data/processed/...")
    df_products = pd.read_csv(processed_dir / "products.csv")
    df_suppliers = pd.read_csv(processed_dir / "suppliers.csv")
    df_warehouses = pd.read_csv(processed_dir / "warehouses.csv")
    df_markets = pd.read_csv(processed_dir / "markets.csv")
    df_date = pd.read_csv(processed_dir / "date.csv")
    df_events = pd.read_csv(processed_dir / "event_log.csv")

    # Add 'NONE' event row if not already present in dim_event to satisfy FK for non-event days
    if "NONE" not in df_events["event_id"].values:
        none_event = pd.DataFrame([{
            "event_id": "NONE",
            "event_name": "Standard Operations (No Active Disruption)",
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "event_type": "Normal",
            "severity": "None",
            "affected_region": "All",
            "affected_categories": "All",
            "affected_suppliers": "All",
            "demand_multiplier": 1.0,
            "lead_time_multiplier": 1.0,
            "transport_cost_multiplier": 1.0,
            "description": "Baseline operational regime without external disruption."
        }])
        df_events = pd.concat([df_events, none_event], ignore_index=True)

    df_products.to_sql("dim_product", conn, if_exists="append", index=False)
    df_suppliers.to_sql("dim_supplier", conn, if_exists="append", index=False)
    df_warehouses.to_sql("dim_warehouse", conn, if_exists="append", index=False)
    df_markets.to_sql("dim_market", conn, if_exists="append", index=False)
    df_date.to_sql("dim_date", conn, if_exists="append", index=False)
    df_events.to_sql("dim_event", conn, if_exists="append", index=False)

    logger.info("Dimension tables populated successfully.")

    # 3. Load Facts
    logger.info("Loading Fact tables from data/processed/...")
    df_demand = pd.read_csv(processed_dir / "demand.csv")
    df_inventory = pd.read_csv(processed_dir / "inventory.csv")
    df_pos = pd.read_csv(processed_dir / "purchase_orders.csv")
    df_transport = pd.read_csv(processed_dir / "transport.csv")

    # Insert chunked for memory efficiency
    logger.info(f"Inserting {len(df_pos):,} purchase orders...")
    df_pos.to_sql("fact_purchase_orders", conn, if_exists="append", index=False, chunksize=5000)

    logger.info(f"Inserting {len(df_transport):,} transport shipments...")
    df_transport.to_sql("fact_transport", conn, if_exists="append", index=False, chunksize=5000)

    logger.info(f"Inserting {len(df_inventory):,} inventory ledger records...")
    df_inventory.to_sql("fact_inventory", conn, if_exists="append", index=False, chunksize=15000)

    logger.info(f"Inserting {len(df_demand):,} customer demand records...")
    df_demand.to_sql("fact_demand", conn, if_exists="append", index=False, chunksize=25000)

    conn.commit()

    # 4. Create Indexes
    logger.info("Building B-Tree indexes on dimension keys and query filters...")
    for idx_stmt in INDEX_STATEMENTS:
        cursor.execute(idx_stmt)
    conn.commit()

    # 5. Referential Integrity Check
    logger.info("Running PRAGMA foreign_key_check to verify relational integrity...")
    cursor.execute("PRAGMA foreign_key_check;")
    fk_violations = cursor.fetchall()
    fk_check_passed = (len(fk_violations) == 0)

    if not fk_check_passed:
        logger.error(f"Foreign key violations detected ({len(fk_violations)} rows): {fk_violations[:5]}")
        raise ValueError(f"Foreign key check failed with {len(fk_violations)} violations")
    else:
        logger.info("Foreign key check PASSED with 0 violations.")

    # 6. Row Counts Validation
    table_counts = {}
    for tbl in [
        "dim_product", "dim_supplier", "dim_warehouse", "dim_market", "dim_date", "dim_event",
        "fact_demand", "fact_inventory", "fact_purchase_orders", "fact_transport"
    ]:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
        cnt = cursor.fetchone()[0]
        table_counts[tbl] = cnt

    conn.close()
    elapsed = round(time.time() - start_time, 2)

    logger.info(f"Database build complete in {elapsed}s.")
    print("\n" + "=" * 60)
    print("DATABASE BUILD COMPLETE")
    print("=" * 60)
    for tbl, count in table_counts.items():
        print(f"{tbl:25}: {count:,} rows")
    print("-" * 60)
    print(f"Foreign key violations   : {len(fk_violations)}")
    print(f"Referential integrity    : {'PASS' if fk_check_passed else 'FAIL'}")
    print(f"Build runtime            : {elapsed} seconds")
    print("=" * 60 + "\n")

    return {
        "database_file": str(db_file),
        "build_runtime_seconds": elapsed,
        "table_counts": table_counts,
        "foreign_key_check_passed": fk_check_passed,
        "foreign_key_violations_count": len(fk_violations)
    }


if __name__ == "__main__":
    build_database()
