"""
Test Suite for Phase 4: SQLite Analytical Database & Star Schema Architecture.
Verifies schema creation, exact row counts against processed CSVs,
strict referential integrity via PRAGMA foreign_key_check, index presence,
analytical query execution, and prevention of raw quality issues in database.
"""

import sqlite3
from pathlib import Path
import pytest
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.database.builder import build_database
from src.database.queries import (
    get_db_connection,
    query_inventory_turnover,
    query_supplier_otif_benchmarks,
    query_service_level_by_category,
    query_cost_breakdown_by_warehouse,
    benchmark_analytical_queries
)


@pytest.fixture(scope="module")
def db_conn():
    root = get_project_root()
    config = load_config()
    db_path = root / config["paths"]["database_file"]
    assert db_path.exists(), f"Database file not found at {db_path}"
    conn = get_db_connection(db_path)
    yield conn
    conn.close()


def test_database_file_exists():
    """Verify operations.db file exists and is populated."""
    root = get_project_root()
    config = load_config()
    db_path = root / config["paths"]["database_file"]
    assert db_path.exists(), "database/operations.db does not exist"
    assert db_path.stat().st_size > 1_000_000, "Database file is unexpectedly small (< 1 MB)"


def test_database_tables_exist(db_conn):
    """Verify all 6 dimension tables and 4 fact tables exist."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = set(r[0] for r in cursor.fetchall())

    expected_dims = [
        "dim_product", "dim_supplier", "dim_warehouse", "dim_market", "dim_date", "dim_event"
    ]
    expected_facts = [
        "fact_demand", "fact_inventory", "fact_purchase_orders", "fact_transport"
    ]

    for d in expected_dims:
        assert d in tables, f"Dimension table '{d}' missing from database"
    for f in expected_facts:
        assert f in tables, f"Fact table '{f}' missing from database"


def test_database_row_counts_match_processed_csvs(db_conn):
    """Verify that SQLite table row counts strictly match data/processed/ certified CSVs."""
    root = get_project_root()
    config = load_config()
    proc_dir = root / config["paths"]["processed_data_dir"]

    mapping = {
        "dim_product": ("products.csv", 60),
        "dim_supplier": ("suppliers.csv", 8),
        "dim_warehouse": ("warehouses.csv", 4),
        "dim_market": ("markets.csv", 6),
        "dim_date": ("date.csv", 731),
        "fact_demand": ("demand.csv", 262978),
        "fact_inventory": ("inventory.csv", 175440),
        "fact_purchase_orders": ("purchase_orders.csv", 11665),
        "fact_transport": ("transport.csv", 11665)
    }

    cursor = db_conn.cursor()
    for table_name, (csv_name, expected_count) in mapping.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        db_count = cursor.fetchone()[0]
        csv_df = pd.read_csv(proc_dir / csv_name)
        assert db_count == len(csv_df), f"Row mismatch for {table_name}: DB={db_count}, CSV={len(csv_df)}"
        assert db_count == expected_count, f"Row count for {table_name} deviated from expected {expected_count}"


def test_foreign_key_referential_integrity(db_conn):
    """
    Run SQLite PRAGMA foreign_key_check to verify zero broken foreign key relationships.
    """
    cursor = db_conn.cursor()
    cursor.execute("PRAGMA foreign_key_check;")
    violations = cursor.fetchall()
    assert len(violations) == 0, f"Foreign key integrity failed with {len(violations)} violations: {violations[:5]}"


def test_indexes_presence(db_conn):
    """Verify composite and single-column performance indexes exist."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = set(r[0] for r in cursor.fetchall())

    expected_indexes = [
        "idx_demand_prod_date",
        "idx_demand_wh_date",
        "idx_demand_market",
        "idx_inv_prod_wh_date",
        "idx_inv_date",
        "idx_inv_wh",
        "idx_po_supplier_date",
        "idx_po_prod_date",
        "idx_po_warehouse",
        "idx_trans_po",
        "idx_trans_dates"
    ]
    for idx in expected_indexes:
        assert idx in indexes, f"Required index '{idx}' not found in database"


def test_analytical_sql_queries(db_conn):
    """Verify execution of analytical queries across the Star Schema."""
    df_turnover = query_inventory_turnover(db_conn)
    assert len(df_turnover) == 24 # 6 categories x 4 warehouses
    assert "annualized_inventory_turnover" in df_turnover.columns

    df_otif = query_supplier_otif_benchmarks(db_conn)
    assert len(df_otif) == 8 # 8 suppliers
    assert "composite_otif_pct" in df_otif.columns
    assert (df_otif["composite_otif_pct"] >= 0).all()

    df_sl = query_service_level_by_category(db_conn)
    assert len(df_sl) == 6 # 6 categories
    assert "service_level_pct" in df_sl.columns

    df_cost = query_cost_breakdown_by_warehouse(db_conn)
    assert len(df_cost) == 4 # 4 warehouses
    assert "total_operational_cost" in df_cost.columns
    assert (df_cost["total_operational_cost"] > 0).all()


def test_no_data_leakage_from_raw_issues(db_conn):
    """Verify that no corrupted identifiers or negative values leaked into the database."""
    cursor = db_conn.cursor()

    # Check no negative demand in fact_demand
    cursor.execute("SELECT COUNT(*) FROM fact_demand WHERE demand_requested < 0;")
    assert cursor.fetchone()[0] == 0, "Negative demand found in fact_demand"

    # Check no orphan 'SKU-999'
    cursor.execute("SELECT COUNT(*) FROM fact_demand WHERE product_id = 'SKU-999';")
    assert cursor.fetchone()[0] == 0, "Orphan product_id SKU-999 found in fact_demand"

    # Check no orphan 'SUP-99'
    cursor.execute("SELECT COUNT(*) FROM fact_purchase_orders WHERE supplier_id = 'SUP-99';")
    assert cursor.fetchone()[0] == 0, "Orphan supplier_id SUP-99 found in fact_purchase_orders"

    # Check no negative ending inventory
    cursor.execute("SELECT COUNT(*) FROM fact_inventory WHERE ending_inventory < 0;")
    assert cursor.fetchone()[0] == 0, "Negative ending inventory found in fact_inventory"


def test_benchmark_performance_thresholds(db_conn):
    """Verify analytical queries meet performance thresholds (< 500ms min or < 1000ms avg)."""
    df_bench = benchmark_analytical_queries(db_conn)
    assert len(df_bench) == 4
    for _, row in df_bench.iterrows():
        assert row["min_latency_ms"] < 500.0 or row["avg_latency_ms"] < 1000.0, f"Query '{row['query_name']}' too slow: {row['avg_latency_ms']}ms"
