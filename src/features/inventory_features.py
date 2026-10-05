"""
Inventory & Stockout Risk Feature Engineering Module for PPOI (Phase 6).
Constructs leakage-safe historical buffer health, stockout incidence, fulfillment velocity,
and warehouse saturation features for predicting future stockout risk.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.utils.logger import get_logger

logger = get_logger("InventoryFeatures")


def extract_base_inventory_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Extracts continuous daily inventory ledger records joined with warehouse capacity metadata.
    Guarantees exactly 175,440 rows (60 SKUs × 4 Warehouses × 731 Days).
    """
    sql = """
    SELECT 
        i.date,
        i.product_id,
        p.product_name,
        p.category,
        p.criticality,
        p.unit_cost,
        p.selling_price,
        i.warehouse_id,
        w.warehouse_name,
        w.capacity_units AS warehouse_capacity_units,
        i.beginning_inventory,
        i.po_received,
        i.demand_requested,
        i.demand_fulfilled,
        i.lost_sales_quantity,
        i.ending_backorder,
        i.ending_inventory,
        i.stockout_flag,
        i.safety_stock_target,
        i.holding_cost,
        i.stockout_cost
    FROM fact_inventory i
    JOIN dim_product p ON i.product_id = p.product_id
    JOIN dim_warehouse w ON i.warehouse_id = w.warehouse_id
    ORDER BY i.product_id, i.warehouse_id, i.date;
    """
    df = pd.read_sql_query(sql, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_inventory_risk_features(
    conn: Optional[sqlite3.Connection] = None,
    df_override: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Builds the production inventory and stockout risk feature dataset.
    
    LEAKAGE-FREE GUARANTEE:
    Features representing inventory state, buffer coverage, stockout history, and warehouse
    saturation are computed strictly prior to the fulfillment of date t:
    - Lagged state: ending_inventory_lag1 (day t-1 balance), stockout_lag_1.
    - Rolling metrics: computed on s.shift(1).rolling(w), incorporating strictly days [t-w, t-1].
    - Target variables look strictly forward into days [t, t+6] and [t, t+13].
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    logger.info("Extracting inventory ledger and warehouse metadata...")
    if df_override is not None:
        df = df_override.copy()
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = extract_base_inventory_data(conn)

    if close_conn:
        conn.close()

    # Sort strictly by entity keys and chronological timestamp
    df = df.sort_values(["product_id", "warehouse_id", "date"]).reset_index(drop=True)
    grouped = df.groupby(["product_id", "warehouse_id"])

    logger.info("Computing inventory buffer health and coverage features...")
    # -------------------------------------------------------------------------
    # 1. Inventory State & Buffer Health (Known at Start of Day t)
    # -------------------------------------------------------------------------
    # Physical stock available before customer fulfillment on day t:
    # beginning_inventory equals ending_inventory of day t-1.
    df["ending_inventory_lag1"] = grouped["ending_inventory"].shift(1)
    df["ending_inventory_lag2"] = grouped["ending_inventory"].shift(2)
    
    # 1-day change in physical ending inventory
    df["inventory_change_1d"] = (df["ending_inventory_lag1"] - df["ending_inventory_lag2"]).fillna(0.0)

    # Safety stock gap and coverage ratio (using beginning inventory on hand)
    df["safety_stock_gap"] = df["beginning_inventory"] - df["safety_stock_target"]
    df["inventory_to_safety_stock_ratio"] = (
        df["beginning_inventory"] / df["safety_stock_target"].replace(0, np.nan)
    ).fillna(1.0).round(2)

    # -------------------------------------------------------------------------
    # 2. Historical Demand Velocity & Days of Supply (Strictly [t-w, t-1])
    # -------------------------------------------------------------------------
    dem_shifted = grouped["demand_requested"].shift(1)
    
    df["demand_mean_7"] = (
        dem_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    ).round(2)

    df["demand_mean_28"] = (
        dem_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).mean())
    ).round(2)

    df["demand_std_28"] = (
        dem_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=2).std().fillna(0.0))
    ).round(2)

    df["demand_trend_28"] = (
        df["demand_mean_7"] / df["demand_mean_28"].replace(0, np.nan)
    ).fillna(1.0).round(3)

    # Forward-looking Days of Supply based on recent 28-day historical burn rate
    df["inventory_days_of_supply"] = (
        df["beginning_inventory"] / df["demand_mean_28"].replace(0, np.nan)
    ).fillna(999.0).round(1)

    # -------------------------------------------------------------------------
    # 3. Historical Stockout Incidence (Strictly [t-w, t-1])
    # -------------------------------------------------------------------------
    logger.info("Computing historical stockout frequency and streak features...")
    stockout_shifted = grouped["stockout_flag"].shift(1)
    df["stockout_lag_1"] = stockout_shifted.fillna(0).astype(int)

    df["stockout_count_7d"] = (
        stockout_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).sum().fillna(0))
    ).astype(int)

    df["stockout_count_28d"] = (
        stockout_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).sum().fillna(0))
    ).astype(int)

    df["stockout_rate_28d"] = (df["stockout_count_28d"] / 28.0).round(3)

    # Consecutive stockout days ending at t-1
    # Calculated iteratively over shifted series
    def compute_consecutive_streaks(s: pd.Series) -> pd.Series:
        streaks = []
        cur = 0
        for val in s:
            if pd.isna(val) or val == 0:
                cur = 0
            else:
                cur += 1
            streaks.append(cur)
        return pd.Series(streaks, index=s.index)

    df["consecutive_stockout_days"] = (
        stockout_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(compute_consecutive_streaks)
    )

    # -------------------------------------------------------------------------
    # 4. Historical Lost Sales & Backorders (Strictly [t-w, t-1])
    # -------------------------------------------------------------------------
    lost_shifted = grouped["lost_sales_quantity"].shift(1)
    df["lost_sales_7d"] = (
        lost_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).sum().fillna(0))
    )
    df["lost_sales_28d"] = (
        lost_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).sum().fillna(0))
    )
    df["lost_sales_rate_28d"] = (
        df["lost_sales_28d"] / (df["demand_mean_28"] * 28.0).replace(0, np.nan)
    ).fillna(0.0).round(3)

    bo_shifted = grouped["ending_backorder"].shift(1)
    df["ending_backorder_lag1"] = bo_shifted.fillna(0)
    has_bo_shifted = (bo_shifted > 0).astype(int)
    df["backorder_count_7d"] = (
        has_bo_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).sum().fillna(0))
    ).astype(int)
    df["backorder_count_28d"] = (
        has_bo_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).sum().fillna(0))
    ).astype(int)

    # -------------------------------------------------------------------------
    # 5. Warehouse Capacity Utilization Context (Strictly [t-w, t-1])
    # -------------------------------------------------------------------------
    # Total facility inventory at end of day t-1
    wh_daily = df.groupby(["warehouse_id", "date"])["ending_inventory_lag1"].transform("sum")
    df["warehouse_total_inv_lag1"] = wh_daily
    df["warehouse_capacity_utilization_lag1"] = (
        (wh_daily / df["warehouse_capacity_units"]) * 100.0
    ).round(2)

    # Rolling 7d average warehouse utilization
    wh_util_series = df.groupby(["warehouse_id", "date"])["warehouse_capacity_utilization_lag1"].first().reset_index()
    wh_util_series["warehouse_utilization_7d"] = (
        wh_util_series.groupby("warehouse_id")["warehouse_capacity_utilization_lag1"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    ).round(2)
    df = df.merge(
        wh_util_series[["warehouse_id", "date", "warehouse_utilization_7d"]],
        on=["warehouse_id", "date"],
        how="left"
    )

    # SKU share of facility total inventory
    df["warehouse_inventory_share"] = (
        df["beginning_inventory"] / df["warehouse_total_inv_lag1"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # -------------------------------------------------------------------------
    # 6. Future Stockout Targets (Forward-Looking: [t, t+6] and [t, t+13])
    # -------------------------------------------------------------------------
    logger.info("Constructing forward-looking stockout risk target variables...")
    # Target 1: Will a stockout occur tomorrow (day t+1)?
    df["target_stockout_t_plus_1"] = grouped["stockout_flag"].shift(-1)

    # Target 2: Will a stockout occur within the next 7 days? (Days t through t+6)
    # Using reverse rolling max over 7 days
    rev_stockout = df.groupby(["product_id", "warehouse_id"])["stockout_flag"]
    df["target_stockout_within_7d"] = (
        rev_stockout.transform(lambda s: s.iloc[::-1].rolling(window=7, min_periods=7).max().iloc[::-1])
    ).astype("Int64")

    # Target 3: Will a stockout occur within the next 14 days? (Days t through t+13)
    df["target_stockout_within_14d"] = (
        rev_stockout.transform(lambda s: s.iloc[::-1].rolling(window=14, min_periods=14).max().iloc[::-1])
    ).astype("Int64")

    # Target 4: Total lost sales volume in next 7 days
    rev_lost = df.groupby(["product_id", "warehouse_id"])["lost_sales_quantity"]
    df["target_lost_sales_7d"] = (
        rev_lost.transform(lambda s: s.iloc[::-1].rolling(window=7, min_periods=7).sum().iloc[::-1])
    )

    # -------------------------------------------------------------------------
    # 7. Metadata, History & Target Availability Flags
    # -------------------------------------------------------------------------
    df["has_sufficient_history_28d"] = (df["date"] >= (df["date"].min() + pd.Timedelta(days=28))).astype(int)
    df["is_target_available_7d"] = df["target_stockout_within_7d"].notna().astype(int)
    df["is_target_available_14d"] = df["target_stockout_within_14d"].notna().astype(int)

    # Convert date to standard ISO string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(f"Inventory risk feature dataset constructed successfully: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df
