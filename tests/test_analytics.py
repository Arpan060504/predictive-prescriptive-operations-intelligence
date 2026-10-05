"""
Test Suite for SQL Analytics, Materialized Views, Reconciliation, & Leakage Invariance (Phase 5).
Ensures source-to-materialized reconciliation, anti-Cartesian protection,
temporal feature leakage prevention, and analytical query integrity.
"""

import sqlite3
import pytest
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.analytics.metrics import KPI_REGISTRY
from src.analytics.views import build_materialized_analytical_tables
from src.analytics.validation import (
    verify_demand_reconciliation,
    verify_inventory_reconciliation,
    verify_cost_summary_reconciliation,
    verify_anti_cartesian_protection,
    verify_cross_phase_demand_reconciliation,
    verify_cross_phase_cost_reconciliation,
    verify_materialized_grains,
    run_full_analytical_audit
)
from src.analytics.feature_base import (
    build_demand_feature_dataset,
    verify_temporal_leakage_invariance
)
from src.analytics.queries import (
    query_executive_kpis,
    query_category_performance,
    query_monthly_cost_breakdown,
    query_supplier_scorecard,
    query_warehouse_capacity_and_utilization,
    query_demand_volatility_and_concentration,
    query_stockout_duration_distribution
)


@pytest.fixture(scope="module")
def db_conn():
    """Provides a shared connection to the operational SQLite database."""
    conn = get_db_connection()
    yield conn
    conn.close()


def test_kpi_registry_integrity():
    """Verifies that all 17 operational KPIs are defined in the registry with complete metadata."""
    assert len(KPI_REGISTRY) >= 17
    for kpi_id, kpi in KPI_REGISTRY.items():
        assert kpi.kpi_id.startswith("KPI-")
        assert len(kpi.name) > 0
        assert len(kpi.business_definition) > 0
        assert len(kpi.mathematical_formula) > 0
        assert len(kpi.source_tables) > 0
        assert len(kpi.sql_implementation) > 0
        assert len(kpi.grain) > 0
        assert len(kpi.unit) > 0


def test_materialized_tables_exist_and_populated(db_conn):
    """Verifies that all 6 physical materialized tables exist and have expected non-zero row counts."""
    expected_tables = {
        "analytics_daily_demand": 175000,
        "analytics_inventory_health": 175440,
        "analytics_supplier_performance": 192,
        "analytics_warehouse_operations": 2924,
        "analytics_cost_summary": 96,
        "analytics_product_supply_risk": 240
    }
    cursor = db_conn.cursor()
    for tbl, min_rows in expected_tables.items():
        cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
        row_cnt = cursor.fetchone()[0]
        assert row_cnt >= min_rows, f"Table {tbl} has {row_cnt} rows, expected at least {min_rows}"


def test_demand_source_reconciliation(db_conn):
    """Verifies that analytics_daily_demand reconciles exactly with fact_demand."""
    res = verify_demand_reconciliation(db_conn)
    assert res["passed"] is True
    assert res["absolute_difference"] == 0


def test_inventory_source_reconciliation(db_conn):
    """Verifies that analytics_inventory_health reconciles exactly with fact_inventory."""
    checks = verify_inventory_reconciliation(db_conn)
    for check in checks:
        assert check["passed"] is True, f"Failed inventory reconciliation: {check}"
        assert check["absolute_difference"] <= 0.05


def test_cost_summary_source_reconciliation(db_conn):
    """Verifies that analytics_cost_summary reconciles exactly with underlying fact tables."""
    checks = verify_cost_summary_reconciliation(db_conn)
    for check in checks:
        assert check["passed"] is True, f"Failed cost reconciliation: {check}"
        assert check["absolute_difference"] <= 0.05


def test_anti_cartesian_protection(db_conn):
    """
    Verifies that CTE pre-aggregation prevents multi-fold Cartesian inflation
    when joining multi-grain operational fact tables.
    """
    res = verify_anti_cartesian_protection(db_conn)
    assert res["passed"] is True
    assert res["cte_matches_truth"] is True
    assert res["naive_cartesian_holding_cost"] > res["true_holding_cost"] * 1000


def test_full_analytical_audit_runner(db_conn):
    """Verifies that the automated audit runner passes all validation checks."""
    audit_summary = run_full_analytical_audit(db_conn)
    assert audit_summary["all_passed"] is True


def test_temporal_feature_leakage_invariance():
    """
    Verifies that feature engineering exhibits strict temporal invariance:
    corrupting future data (after cutoff date) has zero effect on historical features.
    """
    is_invariant = verify_temporal_leakage_invariance(
        build_demand_feature_dataset,
        cutoff_date="2025-06-01"
    )
    assert is_invariant == True


def test_executive_kpis_query(db_conn):
    """Verifies the executive KPIs query output structure, bounds, and consistency."""
    kpis = query_executive_kpis(db_conn)
    assert kpis["entity_counts"]["products"] == 60
    assert kpis["entity_counts"]["suppliers"] == 8
    assert kpis["entity_counts"]["warehouses"] == 4
    assert kpis["entity_counts"]["markets"] == 6

    # Verify service level and stockout frequency bounds
    assert 90.0 <= kpis["operations"]["service_level_pct"] <= 100.0
    assert 5.0 <= kpis["operations"]["stockout_frequency_pct"] <= 15.0
    assert kpis["operations"]["annualized_inventory_turnover"] > 10.0

    # Total operational cost equals sum of 4 components
    fin = kpis["financials"]
    expected_sum = round(
        fin["total_procurement_cost"] + fin["total_holding_cost"] +
        fin["total_stockout_cost"] + fin["total_transport_cost"],
        2
    )
    assert abs(fin["total_operational_cost"] - expected_sum) <= 0.05


def test_supplier_scorecard_percentiles(db_conn):
    """Verifies supplier scorecard includes median and P90 lead time delays."""
    df_sup = query_supplier_scorecard(db_conn)
    assert len(df_sup) == 8
    assert "median_delay_days" in df_sup.columns
    assert "p90_delay_days" in df_sup.columns
    assert (df_sup["median_delay_days"] <= df_sup["p90_delay_days"]).all()
    assert (df_sup["otif_rate_pct"] >= 0.0).all() and (df_sup["otif_rate_pct"] <= 100.0).all()


def test_warehouse_capacity_and_utilization(db_conn):
    """Verifies warehouse capacity utilization aggregations."""
    df_wh = query_warehouse_capacity_and_utilization(db_conn)
    assert len(df_wh) == 4
    assert (df_wh["avg_capacity_utilization_pct"] > 0).all()
    assert (df_wh["peak_capacity_utilization_pct"] >= df_wh["avg_capacity_utilization_pct"]).all()
    assert df_wh["total_lost_sales_units"].sum() == 767368


def test_stockout_duration_distribution(db_conn):
    """Verifies stockout streaks and duration bracket aggregations."""
    df_streaks = query_stockout_duration_distribution(db_conn)
    assert len(df_streaks) == 4
    assert df_streaks["total_stockout_days"].sum() == 14591


def test_cross_phase_demand_reconciliation(db_conn):
    """Verifies cross-phase demand totals between fact_demand and fact_inventory."""
    res = verify_cross_phase_demand_reconciliation(db_conn)
    assert res["passed"] is True
    assert res["fact_demand_sum"] == 19256745
    assert res["fact_inventory_sum"] == 19237514
    assert res["delta"] == 19231


def test_cross_phase_cost_reconciliation(db_conn):
    """Verifies cross-phase cost totals between raw Phase 2 and certified database."""
    res = verify_cross_phase_cost_reconciliation(db_conn)
    assert res["passed"] is True
    assert res["db_total_operational_cost"] == 668327420.34
    assert res["materialized_total_operational_cost"] == 668327420.34
    assert res["quarantined_po_delta"] == 3021901.98


def test_materialized_grains_and_68_missing_days(db_conn):
    """Verifies the exact 68-row difference between daily demand and daily inventory ledger."""
    res = verify_materialized_grains(db_conn)
    assert res["passed"] is True
    assert res["analytics_daily_demand_rows"] == 175372
    assert res["analytics_inventory_health_rows"] == 175440
    assert res["delta_rows"] == 68


def test_executive_kpis_direct_source_reconciliation(db_conn):
    """Verifies that executive KPI layer reconciles directly to SQLite fact tables without leakage."""
    kpis = query_executive_kpis(db_conn)
    cursor = db_conn.cursor()

    # Reconcile costs against fact tables
    cursor.execute("SELECT ROUND(SUM(total_procurement_cost), 2) FROM fact_purchase_orders;")
    src_po = cursor.fetchone()[0]
    cursor.execute("SELECT ROUND(SUM(transport_cost), 2) FROM fact_transport;")
    src_tr = cursor.fetchone()[0]
    cursor.execute("SELECT ROUND(SUM(holding_cost), 2), ROUND(SUM(stockout_cost), 2) FROM fact_inventory;")
    src_hold, src_stock = cursor.fetchone()

    assert kpis["financials"]["total_procurement_cost"] == src_po
    assert kpis["financials"]["total_transport_cost"] == src_tr
    assert kpis["financials"]["total_holding_cost"] == src_hold
    assert kpis["financials"]["total_stockout_cost"] == src_stock

    # Reconcile demand against fact_demand and fact_inventory
    cursor.execute("SELECT SUM(demand_requested) FROM fact_demand;")
    src_comm_dem = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(demand_requested), SUM(demand_fulfilled), SUM(lost_sales_quantity) FROM fact_inventory;")
    src_wh_dem, src_wh_ful, src_wh_lost = cursor.fetchone()

    assert kpis["operations"]["total_commercial_demand_requested"] == src_comm_dem
    assert kpis["operations"]["total_warehouse_demand_requested"] == src_wh_dem
    assert kpis["operations"]["total_demand_fulfilled"] == src_wh_ful
    assert kpis["operations"]["total_lost_sales_units"] == src_wh_lost
