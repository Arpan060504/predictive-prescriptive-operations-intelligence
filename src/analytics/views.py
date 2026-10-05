"""
Analytical SQL Views & Materialized Tables for PPOI.
Defines logical views and materializes high-cost operational datasets into SQLite.
Strictly prevents Cartesian double counting by pre-aggregating individual facts before joining.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("AnalyticsViews")

# -----------------------------------------------------------------------------
# 1. SQL View Definitions (Logical Views)
# -----------------------------------------------------------------------------
SQL_VIEW_DEFINITIONS = [
    # View A: Daily Demand Rollup (Grain: SKU × Warehouse × Date)
    """
    CREATE VIEW IF NOT EXISTS view_daily_demand_rollup AS
    SELECT 
        d.date,
        d.product_id,
        p.product_name,
        p.category,
        d.warehouse_id,
        w.warehouse_name,
        COUNT(DISTINCT d.market_id) AS markets_served_count,
        SUM(d.demand_requested) AS total_demand_requested,
        MAX(d.promotion_flag) AS is_promotional_day,
        MAX(CASE WHEN d.event_id != 'NONE' THEN 1 ELSE 0 END) AS is_disruption_event_day,
        SUM(d.outlier_flag) AS outlier_market_count
    FROM fact_demand d
    JOIN dim_product p ON d.product_id = p.product_id
    JOIN dim_warehouse w ON d.warehouse_id = w.warehouse_id
    GROUP BY d.date, d.product_id, p.product_name, p.category, d.warehouse_id, w.warehouse_name;
    """,

    # View B: Inventory Health Rollup (Grain: SKU × Warehouse × Date)
    """
    CREATE VIEW IF NOT EXISTS view_inventory_health_rollup AS
    SELECT 
        i.date,
        i.product_id,
        p.product_name,
        p.category,
        p.criticality,
        i.warehouse_id,
        w.warehouse_name,
        i.beginning_inventory,
        i.po_received,
        i.demand_requested,
        i.demand_fulfilled,
        i.lost_sales_quantity,
        i.ending_backorder,
        i.ending_inventory,
        i.stockout_flag,
        i.stockout_quantity,
        i.safety_stock_target,
        ROUND(CAST(i.ending_inventory AS REAL) / NULLIF(i.safety_stock_target, 0), 2) AS safety_stock_coverage_ratio,
        i.holding_cost,
        i.stockout_cost
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id;
    """,

    # View C: Supplier Performance Rollup (Grain: Supplier × Month)
    """
    CREATE VIEW IF NOT EXISTS view_supplier_monthly_performance AS
    SELECT 
        SUBSTR(po.order_date, 1, 7) AS year_month,
        s.supplier_id,
        s.supplier_name,
        s.tier,
        COUNT(po.po_id) AS total_orders,
        SUM(po.quantity_ordered) AS total_units_ordered,
        SUM(po.quantity_received) AS total_units_received,
        ROUND(AVG(po.on_time_flag) * 100, 2) AS on_time_rate_pct,
        ROUND(AVG(po.in_full_flag) * 100, 2) AS in_full_rate_pct,
        ROUND(AVG(po.otif_flag) * 100, 2) AS otif_rate_pct,
        ROUND(AVG(po.delay_days), 2) AS avg_delay_days,
        ROUND(SUM(po.total_procurement_cost), 2) AS total_spend,
        ROUND(SUM(po.transport_cost), 2) AS total_freight_cost
    FROM fact_purchase_orders po
    JOIN dim_supplier s ON po.supplier_id = s.supplier_id
    GROUP BY SUBSTR(po.order_date, 1, 7), s.supplier_id, s.supplier_name, s.tier;
    """,

    # View D: Warehouse Operations Daily (Grain: Warehouse × Date)
    """
    CREATE VIEW IF NOT EXISTS view_warehouse_daily_operations AS
    SELECT 
        i.date,
        w.warehouse_id,
        w.warehouse_name,
        w.capacity_units,
        SUM(i.ending_inventory) AS total_ending_inventory,
        ROUND((CAST(SUM(i.ending_inventory) AS REAL) / w.capacity_units) * 100, 2) AS capacity_utilization_pct,
        SUM(i.demand_requested) AS total_demand_requested,
        SUM(i.demand_fulfilled) AS total_demand_fulfilled,
        SUM(i.lost_sales_quantity) AS total_lost_sales,
        SUM(i.stockout_flag) AS stockout_skus_count,
        ROUND(SUM(i.holding_cost), 2) AS daily_holding_cost,
        ROUND(SUM(i.stockout_cost), 2) AS daily_stockout_cost
    FROM fact_inventory i
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
    GROUP BY i.date, w.warehouse_id, w.warehouse_name, w.capacity_units;
    """
]

# -----------------------------------------------------------------------------
# 2. Materialized Tables Creation & Population
# -----------------------------------------------------------------------------
def build_materialized_analytical_tables(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Creates and populates physical analytical tables in SQLite.
    Materializes high-cost aggregations to ensure instantaneous dashboard queries.
    Prevents Cartesian measure inflation by pre-aggregating facts via CTEs.
    """
    cursor = conn.cursor()
    logger.info("Building and materializing analytical datasets in SQLite...")
    
    # 1. Create Views first
    for v_ddl in SQL_VIEW_DEFINITIONS:
        cursor.execute(v_ddl)
    conn.commit()

    # Table 1: analytics_daily_demand (Grain: SKU × Warehouse × Day)
    cursor.execute("DROP TABLE IF EXISTS analytics_daily_demand;")
    cursor.execute("""
    CREATE TABLE analytics_daily_demand AS
    SELECT 
        d.date,
        d.product_id,
        p.product_name,
        p.category,
        d.warehouse_id,
        w.warehouse_name,
        COUNT(DISTINCT d.market_id) AS markets_count,
        SUM(d.demand_requested) AS demand_requested,
        MAX(d.promotion_flag) AS promotion_flag,
        MAX(CASE WHEN d.event_id != 'NONE' THEN 1 ELSE 0 END) AS event_flag,
        MAX(d.outlier_flag) AS outlier_flag
    FROM fact_demand d
    JOIN dim_product p ON d.product_id = p.product_id
    JOIN dim_warehouse w ON d.warehouse_id = w.warehouse_id
    GROUP BY d.date, d.product_id, p.product_name, p.category, d.warehouse_id, w.warehouse_name;
    """)
    cursor.execute("CREATE INDEX idx_mat_dem_pw ON analytics_daily_demand(product_id, warehouse_id, date);")

    # Table 2: analytics_inventory_health (Grain: SKU × Warehouse × Day)
    cursor.execute("DROP TABLE IF EXISTS analytics_inventory_health;")
    cursor.execute("""
    CREATE TABLE analytics_inventory_health AS
    SELECT 
        i.date,
        i.product_id,
        p.product_name,
        p.category,
        p.criticality,
        i.warehouse_id,
        w.warehouse_name,
        i.beginning_inventory,
        i.po_received,
        i.demand_requested,
        i.demand_fulfilled,
        i.lost_sales_quantity,
        i.ending_backorder,
        i.ending_inventory,
        i.stockout_flag,
        i.stockout_quantity,
        i.safety_stock_target,
        ROUND(CAST(i.ending_inventory AS REAL) / NULLIF(i.safety_stock_target, 0), 2) AS safety_stock_coverage,
        i.holding_cost,
        i.stockout_cost
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id;
    """)
    cursor.execute("CREATE INDEX idx_mat_inv_pw ON analytics_inventory_health(product_id, warehouse_id, date);")

    # Table 3: analytics_supplier_performance (Grain: Supplier × Month)
    cursor.execute("DROP TABLE IF EXISTS analytics_supplier_performance;")
    cursor.execute("""
    CREATE TABLE analytics_supplier_performance AS
    SELECT 
        SUBSTR(po.order_date, 1, 7) AS year_month,
        s.supplier_id,
        s.supplier_name,
        s.tier,
        COUNT(po.po_id) AS total_orders,
        SUM(po.quantity_ordered) AS units_ordered,
        SUM(po.quantity_received) AS units_received,
        ROUND(AVG(po.on_time_flag) * 100, 2) AS on_time_rate_pct,
        ROUND(AVG(po.in_full_flag) * 100, 2) AS in_full_rate_pct,
        ROUND(AVG(po.otif_flag) * 100, 2) AS otif_rate_pct,
        ROUND(AVG(po.delay_days), 2) AS avg_delay_days,
        ROUND(SUM(po.total_procurement_cost), 2) AS total_procurement_cost,
        ROUND(SUM(po.transport_cost), 2) AS total_freight_cost
    FROM fact_purchase_orders po
    JOIN dim_supplier s ON po.supplier_id = s.supplier_id
    GROUP BY SUBSTR(po.order_date, 1, 7), s.supplier_id, s.supplier_name, s.tier;
    """)
    cursor.execute("CREATE INDEX idx_mat_sup_sm ON analytics_supplier_performance(supplier_id, year_month);")

    # Table 4: analytics_warehouse_operations (Grain: Warehouse × Day)
    cursor.execute("DROP TABLE IF EXISTS analytics_warehouse_operations;")
    cursor.execute("""
    CREATE TABLE analytics_warehouse_operations AS
    SELECT 
        i.date,
        w.warehouse_id,
        w.warehouse_name,
        w.capacity_units,
        SUM(i.ending_inventory) AS ending_inventory,
        ROUND((CAST(SUM(i.ending_inventory) AS REAL) / w.capacity_units) * 100, 2) AS capacity_utilization_pct,
        SUM(i.demand_requested) AS demand_requested,
        SUM(i.demand_fulfilled) AS demand_fulfilled,
        SUM(i.lost_sales_quantity) AS lost_sales_units,
        SUM(i.stockout_flag) AS stockout_skus_count,
        ROUND(SUM(i.holding_cost), 2) AS holding_cost,
        ROUND(SUM(i.stockout_cost), 2) AS stockout_cost
    FROM fact_inventory i
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
    GROUP BY i.date, w.warehouse_id, w.warehouse_name, w.capacity_units;
    """)
    cursor.execute("CREATE INDEX idx_mat_wh_wd ON analytics_warehouse_operations(warehouse_id, date);")

    # Table 5: analytics_cost_summary (Grain: Warehouse × Month)
    # Strictly prevents Cartesian measure inflation via pre-aggregated CTEs
    cursor.execute("DROP TABLE IF EXISTS analytics_cost_summary;")
    cursor.execute("""
    CREATE TABLE analytics_cost_summary AS
    WITH cte_inv AS (
        SELECT 
            SUBSTR(date, 1, 7) AS year_month,
            warehouse_id,
            SUM(holding_cost) AS total_holding_cost,
            SUM(stockout_cost) AS total_stockout_cost
        FROM fact_inventory
        GROUP BY SUBSTR(date, 1, 7), warehouse_id
    ),
    cte_po AS (
        SELECT 
            SUBSTR(order_date, 1, 7) AS year_month,
            warehouse_id,
            SUM(total_procurement_cost) AS total_procurement_cost,
            SUM(transport_cost) AS total_transport_cost
        FROM fact_purchase_orders
        GROUP BY SUBSTR(order_date, 1, 7), warehouse_id
    )
    SELECT 
        COALESCE(i.year_month, p.year_month) AS year_month,
        w.warehouse_id,
        w.warehouse_name,
        COALESCE(p.total_procurement_cost, 0.0) AS procurement_cost,
        COALESCE(i.total_holding_cost, 0.0) AS holding_cost,
        COALESCE(i.total_stockout_cost, 0.0) AS stockout_cost,
        COALESCE(p.total_transport_cost, 0.0) AS transport_cost,
        (
            COALESCE(p.total_procurement_cost, 0.0) + COALESCE(i.total_holding_cost, 0.0) +
            COALESCE(i.total_stockout_cost, 0.0) + COALESCE(p.total_transport_cost, 0.0)
        ) AS total_operational_cost
    FROM dim_warehouse w
    JOIN cte_inv i ON w.warehouse_id = i.warehouse_id
    LEFT JOIN cte_po p ON w.warehouse_id = p.warehouse_id AND i.year_month = p.year_month;
    """)
    cursor.execute("CREATE INDEX idx_mat_cost_wm ON analytics_cost_summary(warehouse_id, year_month);")

    # Table 6: analytics_product_supply_risk (Grain: SKU × Warehouse)
    cursor.execute("DROP TABLE IF EXISTS analytics_product_supply_risk;")
    cursor.execute("""
    CREATE TABLE analytics_product_supply_risk AS
    SELECT 
        i.product_id,
        p.product_name,
        p.category,
        p.criticality,
        p.primary_supplier_id,
        s.supplier_name AS primary_supplier_name,
        s.baseline_reliability AS supplier_reliability,
        i.warehouse_id,
        w.warehouse_name,
        ROUND(AVG(i.ending_inventory), 1) AS avg_inventory_units,
        ROUND(AVG(i.demand_requested), 1) AS avg_daily_demand,
        ROUND(AVG(i.ending_inventory) / NULLIF(AVG(i.demand_requested), 0), 1) AS avg_days_of_coverage,
        SUM(i.stockout_flag) AS stockout_days_count,
        ROUND((CAST(SUM(i.stockout_flag) AS REAL) / COUNT(*)) * 100, 2) AS stockout_frequency_pct,
        SUM(i.lost_sales_quantity) AS total_lost_sales_units,
        ROUND(
            (CAST(SUM(i.demand_fulfilled) AS REAL) / NULLIF(SUM(i.demand_requested), 0)) * 100,
            2
        ) AS service_level_pct,
        CASE 
            WHEN (CAST(SUM(i.stockout_flag) AS REAL) / COUNT(*)) >= 0.15 OR s.baseline_reliability < 0.85 THEN 'HIGH'
            WHEN (CAST(SUM(i.stockout_flag) AS REAL) / COUNT(*)) >= 0.05 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS empirical_risk_tier
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
    JOIN dim_supplier s ON p.primary_supplier_id = s.supplier_id
    GROUP BY i.product_id, p.product_name, p.category, p.criticality, p.primary_supplier_id, s.supplier_name, s.baseline_reliability, i.warehouse_id, w.warehouse_name;
    """)
    cursor.execute("CREATE INDEX idx_mat_risk_pw ON analytics_product_supply_risk(product_id, warehouse_id);")

    conn.commit()

    # Collect row counts of materialized tables
    mat_tables = [
        "analytics_daily_demand",
        "analytics_inventory_health",
        "analytics_supplier_performance",
        "analytics_warehouse_operations",
        "analytics_cost_summary",
        "analytics_product_supply_risk"
    ]
    counts = {}
    for tbl in mat_tables:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
        counts[tbl] = cursor.fetchone()[0]

    logger.info(f"Materialized analytical tables successfully populated: {counts}")
    return counts
