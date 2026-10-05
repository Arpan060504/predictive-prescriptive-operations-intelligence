"""
Analytical SQL Queries & Performance Benchmarking for PPOI.
Demonstrates analytics engineering by executing complex joins, window aggregations,
and performance benchmarks across the Star Schema SQLite database.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("DatabaseQueries")


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns an optimized SQLite connection with foreign keys enabled."""
    if db_path is None:
        root = get_project_root()
        config = load_config()
        db_path = root / config["paths"]["database_file"]
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def query_inventory_turnover(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Computes annualized inventory turnover, average inventory units,
    and days of inventory coverage (DOIC) by product category and warehouse.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        p.category,
        w.warehouse_id,
        w.warehouse_name,
        COUNT(DISTINCT i.product_id) AS active_skus,
        SUM(i.demand_fulfilled) AS total_fulfilled_units,
        ROUND(AVG(i.ending_inventory), 1) AS avg_daily_inventory_units,
        ROUND(SUM(i.holding_cost), 2) AS total_holding_cost,
        ROUND(
            (CAST(SUM(i.demand_fulfilled) AS REAL) / NULLIF(AVG(i.ending_inventory), 0)) * (365.25 / 731.0),
            2
        ) AS annualized_inventory_turnover,
        ROUND(
            (AVG(i.ending_inventory) / NULLIF(SUM(i.demand_fulfilled) / 731.0, 0)),
            1
        ) AS days_of_inventory_coverage
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
    GROUP BY p.category, w.warehouse_id, w.warehouse_name
    ORDER BY p.category, w.warehouse_id;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def query_supplier_otif_benchmarks(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Evaluates empirical supplier fulfillment performance: total orders,
    on-time delivery rate, in-full delivery rate, composite OTIF rate,
    and average lead-time delay days.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        s.supplier_id,
        s.supplier_name,
        s.tier,
        s.baseline_reliability,
        s.baseline_lead_time_days,
        COUNT(po.po_id) AS total_purchase_orders,
        SUM(po.quantity_ordered) AS total_units_ordered,
        SUM(po.quantity_received) AS total_units_received,
        ROUND(AVG(po.on_time_flag) * 100, 2) AS on_time_delivery_pct,
        ROUND(AVG(po.in_full_flag) * 100, 2) AS in_full_delivery_pct,
        ROUND(AVG(po.otif_flag) * 100, 2) AS composite_otif_pct,
        ROUND(AVG(po.delay_days), 2) AS avg_delay_days,
        ROUND(SUM(po.total_procurement_cost), 2) AS total_procurement_value
    FROM dim_supplier s
    LEFT JOIN fact_purchase_orders po ON s.supplier_id = po.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.tier, s.baseline_reliability, s.baseline_lead_time_days
    ORDER BY composite_otif_pct DESC;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def query_service_level_by_category(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Calculates unit-fill service level, stockout frequency rate,
    and total lost sales units grouped by product category.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        p.category,
        COUNT(DISTINCT i.product_id) AS sku_count,
        SUM(i.demand_requested) AS total_demand_requested,
        SUM(i.demand_fulfilled) AS total_demand_fulfilled,
        SUM(i.lost_sales_quantity) AS total_lost_sales_units,
        ROUND(
            (CAST(SUM(i.demand_fulfilled) AS REAL) / NULLIF(SUM(i.demand_requested), 0)) * 100,
            2
        ) AS service_level_pct,
        SUM(i.stockout_flag) AS stockout_days_count,
        COUNT(*) AS total_sku_wh_days,
        ROUND(
            (CAST(SUM(i.stockout_flag) AS REAL) / COUNT(*)) * 100,
            2
        ) AS stockout_frequency_pct,
        ROUND(SUM(i.stockout_cost), 2) AS total_stockout_penalty_cost
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    GROUP BY p.category
    ORDER BY service_level_pct ASC;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def query_cost_breakdown_by_warehouse(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Aggregates financial cost composition (holding cost, stockout penalties,
    procurement expenditures, and freight transport costs) by regional warehouse.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    WITH wh_inv AS (
        SELECT 
            warehouse_id,
            ROUND(SUM(holding_cost), 2) AS holding_cost,
            ROUND(SUM(stockout_cost), 2) AS stockout_cost
        FROM fact_inventory
        GROUP BY warehouse_id
    ),
    wh_pos AS (
        SELECT 
            warehouse_id,
            ROUND(SUM(total_procurement_cost), 2) AS procurement_cost,
            ROUND(SUM(transport_cost), 2) AS transport_cost
        FROM fact_purchase_orders
        GROUP BY warehouse_id
    )
    SELECT 
        w.warehouse_id,
        w.warehouse_name,
        w.capacity_units,
        COALESCE(p.procurement_cost, 0.0) AS procurement_cost,
        COALESCE(i.holding_cost, 0.0) AS holding_cost,
        COALESCE(i.stockout_cost, 0.0) AS stockout_cost,
        COALESCE(p.transport_cost, 0.0) AS transport_cost,
        ROUND(
            COALESCE(p.procurement_cost, 0.0) + COALESCE(i.holding_cost, 0.0) + 
            COALESCE(i.stockout_cost, 0.0) + COALESCE(p.transport_cost, 0.0),
            2
        ) AS total_operational_cost
    FROM dim_warehouse w
    LEFT JOIN wh_inv i ON w.warehouse_id = i.warehouse_id
    LEFT JOIN wh_pos p ON w.warehouse_id = p.warehouse_id
    ORDER BY total_operational_cost DESC;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def benchmark_analytical_queries(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Benchmarks representative SQL queries across the Star Schema, measuring execution latency.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    queries = [
        ("Inventory Turnover by Category/Warehouse", query_inventory_turnover),
        ("Supplier OTIF & Delay Benchmarks", query_supplier_otif_benchmarks),
        ("Service Level & Stockout Exposure", query_service_level_by_category),
        ("Warehouse Cost Breakdown & Composition", query_cost_breakdown_by_warehouse)
    ]

    results = []
    for name, func in queries:
        latencies = []
        for _ in range(3): # Run 3 iterations
            t0 = time.perf_counter()
            df = func(conn)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
        avg_ms = round(float(sum(latencies) / len(latencies)), 2)
        min_ms = round(float(min(latencies)), 2)
        results.append({
            "query_name": name,
            "rows_returned": len(df),
            "avg_latency_ms": avg_ms,
            "min_latency_ms": min_ms,
            "status": "OPTIMIZED (< 250ms)" if avg_ms < 250 else "ACCEPTABLE"
        })

    if close_conn:
        conn.close()

    df_res = pd.DataFrame(results)
    return df_res
