"""
Analytical Queries & Reporting Helpers for PPOI (Phase 5).
Provides optimized analytical SQL queries querying materialized analytical tables
and views, returning clean pandas DataFrames and dictionaries for dashboarding and models.
"""

import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.utils.logger import get_logger

logger = get_logger("AnalyticsQueries")


def query_executive_kpis(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """
    Computes global enterprise-level executive summary KPIs across the network.
    Queries materialized analytical tables for sub-millisecond aggregation.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    # 1. Total Financial Metrics from analytics_cost_summary
    sql_costs = """
    SELECT 
        ROUND(SUM(procurement_cost), 2) AS total_procurement_cost,
        ROUND(SUM(holding_cost), 2) AS total_holding_cost,
        ROUND(SUM(stockout_cost), 2) AS total_stockout_cost,
        ROUND(SUM(transport_cost), 2) AS total_transport_cost,
        ROUND(SUM(total_operational_cost), 2) AS total_operational_cost
    FROM analytics_cost_summary;
    """
    cost_row = pd.read_sql_query(sql_costs, conn).iloc[0].to_dict()

    # 2. Demand & Fulfillment: Commercial Demand vs. Warehouse Operational Demand
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(demand_requested) FROM analytics_daily_demand;")
    total_commercial_demand = cursor.fetchone()[0] or 0

    sql_service = """
    SELECT 
        SUM(demand_requested) AS total_warehouse_demand_requested,
        SUM(demand_fulfilled) AS total_demand_fulfilled,
        SUM(lost_sales_quantity) AS total_lost_sales_units,
        ROUND(CAST(SUM(demand_fulfilled) AS REAL) / NULLIF(SUM(demand_requested), 0) * 100, 4) AS service_level_pct,
        COUNT(*) AS total_sku_warehouse_days,
        SUM(stockout_flag) AS stockout_sku_warehouse_days,
        ROUND(CAST(SUM(stockout_flag) AS REAL) / COUNT(*) * 100, 2) AS stockout_frequency_pct,
        ROUND(AVG(ending_inventory), 1) AS avg_network_inventory_units
    FROM analytics_inventory_health;
    """
    service_row = pd.read_sql_query(sql_service, conn).iloc[0].to_dict()

    # 3. Supplier Performance & OTIF from analytics_supplier_performance
    sql_supplier = """
    SELECT 
        SUM(total_orders) AS total_purchase_orders,
        SUM(units_ordered) AS total_units_ordered,
        SUM(units_received) AS total_units_received,
        ROUND(SUM(units_received) / CAST(SUM(units_ordered) AS REAL) * 100, 2) AS quantity_fill_rate_pct,
        ROUND(AVG(otif_rate_pct), 2) AS avg_supplier_otif_pct,
        ROUND(AVG(avg_delay_days), 2) AS network_avg_delay_days
    FROM analytics_supplier_performance;
    """
    supplier_row = pd.read_sql_query(sql_supplier, conn).iloc[0].to_dict()

    # 4. Inventory Turnover (Annualized: COGS / Avg Daily Network Inventory * (365.25 / 731))
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(ending_inventory) FROM analytics_inventory_health;")
    total_network_inv_sum = cursor.fetchone()[0] or 0.0
    avg_daily_network_inv = total_network_inv_sum / 731.0
    tot_ful = service_row["total_demand_fulfilled"]
    turnover = round((tot_ful / avg_daily_network_inv) * (365.25 / 731.0), 2) if avg_daily_network_inv > 0 else 0.0

    # 5. Entity Counts
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM dim_product;")
    total_products = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM dim_supplier;")
    total_suppliers = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM dim_warehouse;")
    total_warehouses = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM dim_market;")
    total_markets = cursor.fetchone()[0]

    if close_conn:
        conn.close()

    return {
        "entity_counts": {
            "products": total_products,
            "suppliers": total_suppliers,
            "warehouses": total_warehouses,
            "markets": total_markets,
        },
        "financials": {
            "total_procurement_cost": cost_row["total_procurement_cost"],
            "total_holding_cost": cost_row["total_holding_cost"],
            "total_stockout_cost": cost_row["total_stockout_cost"],
            "total_transport_cost": cost_row["total_transport_cost"],
            "total_operational_cost": cost_row["total_operational_cost"],
        },
        "operations": {
            "total_commercial_demand_requested": total_commercial_demand,
            "total_warehouse_demand_requested": service_row["total_warehouse_demand_requested"],
            "total_demand_requested": service_row["total_warehouse_demand_requested"],
            "total_demand_fulfilled": service_row["total_demand_fulfilled"],
            "total_lost_sales_units": service_row["total_lost_sales_units"],
            "service_level_pct": round(service_row["service_level_pct"], 2),
            "stockout_frequency_pct": service_row["stockout_frequency_pct"],
            "total_sku_warehouse_days": service_row["total_sku_warehouse_days"],
            "stockout_sku_warehouse_days": service_row["stockout_sku_warehouse_days"],
            "annualized_inventory_turnover": turnover,
        },
        "procurement": {
            "total_purchase_orders": supplier_row["total_purchase_orders"],
            "total_units_ordered": supplier_row["total_units_ordered"],
            "total_units_received": supplier_row["total_units_received"],
            "supplier_otif_pct": supplier_row["avg_supplier_otif_pct"],
            "supplier_avg_delay_days": supplier_row["network_avg_delay_days"],
        }
    }


def query_category_performance(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Computes volumetric, service level, and financial breakdown by product category.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        category,
        COUNT(DISTINCT product_id) AS sku_count,
        SUM(demand_requested) AS total_demand_requested,
        SUM(demand_fulfilled) AS total_demand_fulfilled,
        SUM(lost_sales_quantity) AS total_lost_sales_units,
        ROUND(CAST(SUM(demand_fulfilled) AS REAL) / NULLIF(SUM(demand_requested), 0) * 100, 2) AS service_level_pct,
        SUM(stockout_flag) AS stockout_sku_days,
        ROUND(CAST(SUM(stockout_flag) AS REAL) / COUNT(*) * 100, 2) AS stockout_frequency_pct,
        ROUND(AVG(ending_inventory), 1) AS avg_inventory_units,
        ROUND(SUM(holding_cost), 2) AS total_holding_cost,
        ROUND(SUM(stockout_cost), 2) AS total_stockout_cost
    FROM analytics_inventory_health
    GROUP BY category
    ORDER BY total_demand_requested DESC;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def query_monthly_cost_breakdown(
    conn: Optional[sqlite3.Connection] = None,
    warehouse_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Monthly time-series of procurement, holding, stockout, and transport costs.
    Optionally filters by warehouse_id.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    if warehouse_id:
        sql = """
        SELECT 
            year_month,
            warehouse_id,
            warehouse_name,
            procurement_cost,
            holding_cost,
            stockout_cost,
            transport_cost,
            total_operational_cost
        FROM analytics_cost_summary
        WHERE warehouse_id = ?
        ORDER BY year_month;
        """
        df = pd.read_sql_query(sql, conn, params=[warehouse_id])
    else:
        sql = """
        SELECT 
            year_month,
            ROUND(SUM(procurement_cost), 2) AS procurement_cost,
            ROUND(SUM(holding_cost), 2) AS holding_cost,
            ROUND(SUM(stockout_cost), 2) AS stockout_cost,
            ROUND(SUM(transport_cost), 2) AS transport_cost,
            ROUND(SUM(total_operational_cost), 2) AS total_operational_cost
        FROM analytics_cost_summary
        GROUP BY year_month
        ORDER BY year_month;
        """
        df = pd.read_sql_query(sql, conn)

    if close_conn:
        conn.close()
    return df


def query_supplier_scorecard(
    conn: Optional[sqlite3.Connection] = None,
    supplier_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes comprehensive supplier scorecard including Mean, Median, and P90 lead time delays.
    Combines aggregated metrics from analytics_supplier_performance with percentile calculations.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    # 1. Base aggregations from materialized summary
    filter_clause = "WHERE s.supplier_id = ?" if supplier_id else ""
    params = [supplier_id] if supplier_id else []

    sql_summary = f"""
    SELECT 
        s.supplier_id,
        s.supplier_name,
        s.tier,
        s.baseline_lead_time_days AS contractual_lead_time_days,
        s.baseline_reliability,
        SUM(p.total_orders) AS total_orders,
        SUM(p.units_ordered) AS units_ordered,
        SUM(p.units_received) AS units_received,
        ROUND(AVG(p.on_time_rate_pct), 2) AS on_time_rate_pct,
        ROUND(AVG(p.in_full_rate_pct), 2) AS in_full_rate_pct,
        ROUND(AVG(p.otif_rate_pct), 2) AS otif_rate_pct,
        ROUND(AVG(p.avg_delay_days), 2) AS avg_delay_days,
        ROUND(SUM(p.total_procurement_cost), 2) AS total_spend,
        ROUND(SUM(p.total_freight_cost), 2) AS total_freight_cost
    FROM dim_supplier s
    LEFT JOIN analytics_supplier_performance p ON s.supplier_id = p.supplier_id
    {filter_clause}
    GROUP BY s.supplier_id, s.supplier_name, s.tier, s.baseline_lead_time_days, s.baseline_reliability
    ORDER BY total_spend DESC;
    """
    df_summary = pd.read_sql_query(sql_summary, conn, params=params)

    # 2. Compute exact delay distribution percentiles from fact_purchase_orders
    sql_delays = f"""
    SELECT supplier_id, delay_days 
    FROM fact_purchase_orders
    {"WHERE supplier_id = ?" if supplier_id else ""}
    """
    df_delays = pd.read_sql_query(sql_delays, conn, params=params)

    if not df_delays.empty:
        delay_stats = df_delays.groupby("supplier_id")["delay_days"].agg(
            median_delay_days=lambda x: np.percentile(x, 50),
            p90_delay_days=lambda x: np.percentile(x, 90)
        ).reset_index()

        df_summary = df_summary.merge(delay_stats, on="supplier_id", how="left")
    else:
        df_summary["median_delay_days"] = 0.0
        df_summary["p90_delay_days"] = 0.0

    if close_conn:
        conn.close()
    return df_summary


def query_warehouse_capacity_and_utilization(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Computes capacity utilization, throughput, and lost sales by distribution center.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        warehouse_id,
        warehouse_name,
        capacity_units,
        ROUND(AVG(ending_inventory), 1) AS avg_ending_inventory,
        MAX(ending_inventory) AS peak_ending_inventory,
        ROUND(AVG(capacity_utilization_pct), 2) AS avg_capacity_utilization_pct,
        ROUND(MAX(capacity_utilization_pct), 2) AS peak_capacity_utilization_pct,
        SUM(demand_requested) AS total_demand_requested,
        SUM(demand_fulfilled) AS total_demand_fulfilled,
        SUM(lost_sales_units) AS total_lost_sales_units,
        SUM(stockout_skus_count) AS total_stockout_events,
        ROUND(SUM(holding_cost), 2) AS total_holding_cost,
        ROUND(SUM(stockout_cost), 2) AS total_stockout_cost
    FROM analytics_warehouse_operations
    GROUP BY warehouse_id, warehouse_name, capacity_units
    ORDER BY warehouse_id;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()
    return df


def query_demand_volatility_and_concentration(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Computes demand volatility (Coefficient of Variation) and market geographic concentration
    (Herfindahl-Hirschman Index / HHI) for each product across all demand markets.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    # 1. Product-level daily volatility
    sql_volatility = """
    SELECT 
        product_id,
        product_name,
        category,
        ROUND(AVG(demand_requested), 2) AS mean_daily_demand,
        ROUND(
            SQRT(AVG(demand_requested * demand_requested) - AVG(demand_requested) * AVG(demand_requested)), 
            2
        ) AS std_daily_demand,
        SUM(demand_requested) AS total_product_demand
    FROM analytics_daily_demand
    GROUP BY product_id, product_name, category;
    """
    df_vol = pd.read_sql_query(sql_volatility, conn)
    df_vol["demand_cv"] = (
        df_vol["std_daily_demand"] / df_vol["mean_daily_demand"].replace(0, np.nan)
    ).fillna(0.0).round(3)

    # 2. Market geographic concentration (HHI) from fact_demand
    sql_mkt = """
    SELECT 
        product_id,
        market_id,
        SUM(demand_requested) AS market_demand
    FROM fact_demand
    GROUP BY product_id, market_id;
    """
    df_mkt = pd.read_sql_query(sql_mkt, conn)

    # Compute market share squared and HHI per product
    product_totals = df_mkt.groupby("product_id")["market_demand"].transform("sum")
    df_mkt["market_share_pct"] = (df_mkt["market_demand"] / product_totals) * 100
    df_mkt["share_sq"] = df_mkt["market_share_pct"] ** 2

    df_hhi = df_mkt.groupby("product_id")["share_sq"].sum().round(1).reset_index()
    df_hhi.rename(columns={"share_sq": "market_hhi"}, inplace=True)

    df_result = df_vol.merge(df_hhi, on="product_id", how="left")
    df_result = df_result.sort_values("total_product_demand", ascending=False).reset_index(drop=True)

    if close_conn:
        conn.close()
    return df_result


def query_stockout_duration_distribution(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Analyzes consecutive stockout streaks across SKU-warehouse-days.
    Returns distribution of stockout duration lengths (1 day, 2 days, 3-5 days, 6+ days).
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    sql = """
    SELECT 
        date,
        product_id,
        warehouse_id,
        stockout_flag
    FROM analytics_inventory_health
    ORDER BY product_id, warehouse_id, date;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()

    # Identify consecutive streaks of stockout_flag == 1
    df["stockout_change"] = (
        (df["stockout_flag"] != df.groupby(["product_id", "warehouse_id"])["stockout_flag"].shift(1))
        .astype(int)
    )
    df["streak_id"] = df.groupby(["product_id", "warehouse_id"])["stockout_change"].cumsum()

    # Filter to stockout periods only
    stockouts_only = df[df["stockout_flag"] == 1]
    streak_lengths = stockouts_only.groupby(
        ["product_id", "warehouse_id", "streak_id"]
    ).size().reset_index(name="streak_days")

    def categorize_streak(days: int) -> str:
        if days == 1:
            return "1 Day (Transient)"
        elif days == 2:
            return "2 Days (Short)"
        elif 3 <= days <= 5:
            return "3-5 Days (Moderate)"
        else:
            return "6+ Days (Severe/Disrupted)"

    streak_lengths["duration_bracket"] = streak_lengths["streak_days"].apply(categorize_streak)
    bracket_summary = streak_lengths.groupby("duration_bracket").agg(
        episode_count=("streak_days", "count"),
        total_stockout_days=("streak_days", "sum"),
        avg_streak_days=("streak_days", "mean"),
        max_streak_days=("streak_days", "max")
    ).reset_index()

    total_episodes = bracket_summary["episode_count"].sum()
    bracket_summary["pct_of_episodes"] = (bracket_summary["episode_count"] / total_episodes * 100).round(2)
    bracket_summary["avg_streak_days"] = bracket_summary["avg_streak_days"].round(2)

    return bracket_summary.sort_values("total_stockout_days", ascending=False).reset_index(drop=True)
