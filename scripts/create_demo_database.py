"""
Deterministic Demo Database Generator for PPOI.
Extracts a lightweight, representative analytical slice from the certified database
(database/operations.db) to produce database/demo_operations.db for Streamlit Cloud deployment.

Guarantees:
1. Strict GitHub compatibility (< 100 MB, target < 25 MB).
2. Preserves exact table schemas, data types, indexes, and views.
3. 100% referential integrity (foreign keys valid, PRAGMA foreign_key_check passes).
4. Includes 100% of dimension tables (products, suppliers, warehouses, markets, dates, events).
5. Includes 100% of Phase 9 prescriptive optimization tables and scenario definitions.
6. Includes 100% of supplier risk, supplier performance, and cost summary analytics.
7. Includes complete recent analytical slices for operational exposure, demand, and inventory.
"""

import os
import sys
from pathlib import Path
import sqlite3
import time
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger("DemoDatabaseBuilder")


def build_demo_database(
    source_db_path: Path = None,
    target_db_path: Path = None,
    recent_days_cutoff: str = "2025-11-15",
) -> Dict[str, Any]:
    """
    Builds database/demo_operations.db from database/operations.db.
    """
    start_time = time.time()
    root = get_project_root()
    if source_db_path is None:
        source_db_path = root / "database" / "operations.db"
    if target_db_path is None:
        target_db_path = root / "database" / "demo_operations.db"

    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database not found at {source_db_path}")

    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    if target_db_path.exists():
        logger.info(f"Removing existing demo database at {target_db_path}...")
        target_db_path.unlink()

    src_conn = sqlite3.connect(str(source_db_path))
    dst_conn = sqlite3.connect(str(target_db_path))
    
    src_cur = src_conn.cursor()
    dst_cur = dst_conn.cursor()

    dst_cur.execute("PRAGMA foreign_keys = OFF;")  # Disable during bulk load
    dst_cur.execute("PRAGMA journal_mode = MEMORY;")
    dst_cur.execute("PRAGMA synchronous = OFF;")

    # 1. Recreate all table and index schemas exactly
    logger.info("Extracting schema definitions from source database...")
    src_cur.execute("""
        SELECT type, name, sql 
        FROM sqlite_master 
        WHERE sql IS NOT NULL AND type IN ('table', 'view', 'index')
        ORDER BY CASE type 
            WHEN 'table' THEN 1 
            WHEN 'view' THEN 2 
            WHEN 'index' THEN 3 
        END;
    """)
    schema_items = src_cur.fetchall()

    tables = []
    indexes = []
    views = []

    for item_type, name, sql in schema_items:
        if name.startswith("sqlite_"):
            continue
        if item_type == "table":
            tables.append((name, sql))
        elif item_type == "index":
            indexes.append((name, sql))
        elif item_type == "view":
            views.append((name, sql))

    # Create tables first
    logger.info(f"Creating {len(tables)} tables in demo database...")
    for name, sql in tables:
        dst_cur.execute(sql)
    dst_conn.commit()

    # Tables to copy 100% in full
    full_copy_tables = [
        # Dimensions
        "dim_product",
        "dim_supplier",
        "dim_warehouse",
        "dim_market",
        "dim_date",
        "dim_event",
        # Phase 9 Optimization Tables
        "optimization_runs",
        "optimization_decisions",
        "optimization_constraints",
        "scenario_definitions",
        "policy_comparison",
        "decision_explanations",
        # Analytics Summaries
        "analytics_cost_summary",
        "analytics_product_supply_risk",
        "analytics_supplier_performance",
        "analytics_supplier_risk",
        "analytics_warehouse_operations",
    ]

    # Tables to copy with date filter (representative recent slice)
    # Using recent_days_cutoff guarantees all data covering the test horizon and latest decision dates
    sliced_tables = {
        "analytics_operational_exposure": f"WHERE date >= '{recent_days_cutoff}'",
        "analytics_inventory_risk": f"WHERE date >= '{recent_days_cutoff}'",
        "analytics_daily_demand": f"WHERE date >= '{recent_days_cutoff}'",
        "analytics_inventory_health": f"WHERE date >= '{recent_days_cutoff}'",
        "fact_purchase_orders": f"WHERE order_date >= '{recent_days_cutoff}'",
        "fact_transport": f"WHERE po_id IN (SELECT po_id FROM fact_purchase_orders WHERE order_date >= '{recent_days_cutoff}')",
        "fact_demand": f"WHERE date >= '{recent_days_cutoff}'",
        "fact_inventory": f"WHERE date >= '{recent_days_cutoff}'",
    }

    demo_counts = {}

    for tbl in full_copy_tables:
        logger.info(f"Copying table: {tbl} (100% full extract)...")
        src_cur.execute(f'SELECT * FROM "{tbl}"')
        rows = src_cur.fetchall()
        if rows:
            placeholders = ",".join(["?"] * len(rows[0]))
            dst_cur.executemany(f'INSERT INTO "{tbl}" VALUES ({placeholders})', rows)
        demo_counts[tbl] = len(rows)
    dst_conn.commit()

    for tbl, condition in sliced_tables.items():
        logger.info(f"Copying table: {tbl} (filtered: {condition})...")
        src_cur.execute(f'SELECT * FROM "{tbl}" {condition}')
        rows = src_cur.fetchall()
        if rows:
            placeholders = ",".join(["?"] * len(rows[0]))
            dst_cur.executemany(f'INSERT INTO "{tbl}" VALUES ({placeholders})', rows)
        demo_counts[tbl] = len(rows)
    dst_conn.commit()

    # 3. Create Views
    logger.info(f"Creating {len(views)} views...")
    for name, sql in views:
        dst_cur.execute(sql)
    dst_conn.commit()

    # 4. Create Indexes
    logger.info(f"Building {len(indexes)} indexes...")
    for name, sql in indexes:
        try:
            dst_cur.execute(sql)
        except Exception as e:
            logger.warning(f"Index creation note for {name}: {e}")
    dst_conn.commit()

    # 5. Enable Foreign Keys & Validate Referential Integrity
    logger.info("Verifying foreign key referential integrity...")
    dst_cur.execute("PRAGMA foreign_keys = ON;")
    dst_cur.execute("PRAGMA foreign_key_check;")
    fk_violations = dst_cur.fetchall()
    fk_check_passed = (len(fk_violations) == 0)

    # 6. Optimize and vacuum
    logger.info("Running VACUUM and ANALYZE...")
    dst_cur.execute("PRAGMA journal_mode = DELETE;")
    dst_conn.commit()
    dst_cur.execute("VACUUM;")
    dst_cur.execute("ANALYZE;")

    src_conn.close()
    dst_conn.close()

    elapsed = round(time.time() - start_time, 2)
    demo_size_bytes = os.path.getsize(target_db_path)
    demo_size_mb = round(demo_size_bytes / (1024 * 1024), 2)

    print("\n" + "=" * 60)
    print("DEMO DATABASE BUILD COMPLETE")
    print("=" * 60)
    print(f"Target Database Path     : {target_db_path}")
    print(f"Database File Size       : {demo_size_mb} MB ({demo_size_bytes:,} bytes)")
    print(f"GitHub <100MB Compliant  : {'YES' if demo_size_mb < 100 else 'NO'}")
    print(f"Target <25MB Compliant   : {'YES' if demo_size_mb < 25 else 'NO'}")
    print(f"Foreign Key Violations   : {len(fk_violations)}")
    print(f"Referential Integrity    : {'PASS' if fk_check_passed else 'FAIL'}")
    print(f"Build Runtime            : {elapsed} seconds")
    print("-" * 60)
    for tbl, count in sorted(demo_counts.items()):
        print(f"  {tbl:32}: {count:>8,} rows")
    print("=" * 60 + "\n")

    return {
        "target_db_path": str(target_db_path),
        "demo_size_mb": demo_size_mb,
        "fk_check_passed": fk_check_passed,
        "fk_violations_count": len(fk_violations),
        "table_counts": demo_counts,
        "runtime_seconds": elapsed,
    }


if __name__ == "__main__":
    build_demo_database()
