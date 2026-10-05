"""
Demand Forecasting Feature Engineering Module for PPOI (Phase 6).
Constructs leakage-safe historical lag, rolling, dynamic, calendar, and event features
for predicting SKU-Warehouse-Day future customer demand.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.utils.logger import get_logger

logger = get_logger("DemandFeatures")


def extract_base_demand_grid(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Extracts a complete, unbroken daily grid (dim_date × dim_product × dim_warehouse)
    joined with analytics_daily_demand to guarantee 731 contiguous calendar days per SKU-warehouse.
    Days with zero orders are assigned demand_requested = 0.
    """
    sql = """
    SELECT 
        d.date,
        p.product_id,
        p.product_name,
        p.category,
        w.warehouse_id,
        w.warehouse_name,
        COALESCE(dem.demand_requested, 0) AS demand_requested,
        COALESCE(dem.promotion_flag, 0) AS promotion_flag,
        COALESCE(dem.event_flag, 0) AS event_flag,
        COALESCE(dem.outlier_flag, 0) AS outlier_flag
    FROM dim_date d
    CROSS JOIN dim_product p
    CROSS JOIN dim_warehouse w
    LEFT JOIN analytics_daily_demand dem 
        ON d.date = dem.date 
        AND p.product_id = dem.product_id 
        AND w.warehouse_id = dem.warehouse_id
    ORDER BY p.product_id, w.warehouse_id, d.date;
    """
    df = pd.read_sql_query(sql, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_event_metadata(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extracts operational disruption and promotional event windows from dim_event."""
    sql = """
    SELECT 
        event_id,
        event_name,
        start_date,
        end_date,
        event_type,
        severity,
        demand_multiplier,
        lead_time_multiplier,
        transport_cost_multiplier
    FROM dim_event
    WHERE event_id != 'NONE';
    """
    df_events = pd.read_sql_query(sql, conn)
    df_events["start_date"] = pd.to_datetime(df_events["start_date"])
    df_events["end_date"] = pd.to_datetime(df_events["end_date"])
    return df_events


def compute_demand_features(
    conn: Optional[sqlite3.Connection] = None,
    df_override: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Builds the production demand forecasting feature dataset.
    
    LEAKAGE-FREE GUARANTEE:
    All historical features are calculated strictly using observations available prior to day t:
    - Lag features use s.shift(k) where k >= 1.
    - Rolling window statistics are computed on s.shift(1).rolling(w), incorporating strictly
      days [t-w, t-1].
    - Forecast target columns look strictly forward (t+1, t+7, t+14, t+28).
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    logger.info("Extracting continuous demand grid across all SKUs, facilities, and calendar dates...")
    if df_override is not None:
        df = df_override.copy()
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = extract_base_demand_grid(conn)

    df_events = extract_event_metadata(conn)
    if close_conn:
        conn.close()

    # Sort strictly by entity keys and chronological timestamp
    df = df.sort_values(["product_id", "warehouse_id", "date"]).reset_index(drop=True)
    grouped = df.groupby(["product_id", "warehouse_id"])

    logger.info("Computing leakage-safe historical lag and rolling features...")
    # -------------------------------------------------------------------------
    # 1. Historical Lags (strictly t - lag)
    # -------------------------------------------------------------------------
    df["demand_lag_1"] = grouped["demand_requested"].shift(1)
    df["demand_lag_2"] = grouped["demand_requested"].shift(2)
    df["demand_lag_7"] = grouped["demand_requested"].shift(7)
    df["demand_lag_14"] = grouped["demand_requested"].shift(14)
    df["demand_lag_28"] = grouped["demand_requested"].shift(28)

    # -------------------------------------------------------------------------
    # 2. Shifted Rolling Windows (strictly [t-w, t-1])
    # -------------------------------------------------------------------------
    shifted = grouped["demand_requested"].shift(1)
    
    for w in [7, 14, 28]:
        df[f"rolling_mean_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        ).round(2)

    for w in [7, 28]:
        df[f"rolling_std_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .transform(lambda s: s.rolling(window=w, min_periods=2).std().fillna(0.0))
        ).round(2)

    df["rolling_min_28"] = (
        shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).min())
    )
    df["rolling_max_28"] = (
        shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).max())
    )

    # -------------------------------------------------------------------------
    # 3. Demand Dynamics & Volatility Metrics
    # -------------------------------------------------------------------------
    df["demand_change_1d"] = (df["demand_lag_1"] - df["demand_lag_2"]).fillna(0.0)
    df["demand_change_7d"] = (df["demand_lag_1"] - df["demand_lag_7"]).fillna(0.0)
    
    # Growth ratio: rolling 7d mean / rolling 28d mean
    df["demand_growth_28d"] = (
        df["rolling_mean_7"] / df["rolling_mean_28"].replace(0, np.nan)
    ).fillna(1.0).round(3)

    # Demand Coefficient of Variation over historical 28 days
    df["demand_cv_28"] = (
        df["rolling_std_28"] / df["rolling_mean_28"].replace(0, np.nan)
    ).fillna(0.0).round(3)

    # Zero-demand frequency: fraction of days with 0 demand in [t-28, t-1]
    is_zero_shifted = (shifted == 0).astype(int)
    df["zero_demand_frequency_28"] = (
        is_zero_shifted.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.rolling(window=28, min_periods=1).mean())
    ).round(3)

    # -------------------------------------------------------------------------
    # 4. Deterministic Calendar Features (known in advance at t)
    # -------------------------------------------------------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month_start"] = df["date"].dt.is_month_start.astype(int)
    df["month_end"] = df["date"].dt.is_month_end.astype(int)

    # Seasonal Holiday Flag (New Year, Memorial Day, Labor Day, Black Friday, Christmas)
    holiday_dates = pd.to_datetime([
        "2024-01-01", "2024-05-27", "2024-07-04", "2024-09-02", "2024-11-29", "2024-12-25",
        "2025-01-01", "2025-05-26", "2025-07-04", "2025-09-01", "2025-11-28", "2025-12-25"
    ])
    df["holiday_flag"] = df["date"].isin(holiday_dates).astype(int)

    # -------------------------------------------------------------------------
    # 5. Business & Event Features (known at date t)
    # -------------------------------------------------------------------------
    df["event_active"] = 0
    df["event_type"] = "Normal"
    df["event_severity"] = "None"
    df["event_demand_multiplier"] = 1.0
    df["event_lead_time_multiplier"] = 1.0
    df["event_transport_multiplier"] = 1.0
    df["days_since_event_start"] = -1
    df["days_until_event_end"] = -1

    # Vectorized event matching
    for _, ev in df_events.iterrows():
        mask = (df["date"] >= ev["start_date"]) & (df["date"] <= ev["end_date"])
        df.loc[mask, "event_active"] = 1
        df.loc[mask, "event_type"] = ev["event_type"]
        df.loc[mask, "event_severity"] = ev["severity"]
        df.loc[mask, "event_demand_multiplier"] = float(ev["demand_multiplier"])
        df.loc[mask, "event_lead_time_multiplier"] = float(ev["lead_time_multiplier"])
        df.loc[mask, "event_transport_multiplier"] = float(ev["transport_cost_multiplier"])
        df.loc[mask, "days_since_event_start"] = (df.loc[mask, "date"] - ev["start_date"]).dt.days
        df.loc[mask, "days_until_event_end"] = (ev["end_date"] - df.loc[mask, "date"]).dt.days

    # -------------------------------------------------------------------------
    # 6. Future Forecast Targets (Strictly t + horizon)
    # -------------------------------------------------------------------------
    logger.info("Constructing forward-looking forecast target variables...")
    # Single-step horizons
    df["target_demand_t_plus_1"] = grouped["demand_requested"].shift(-1)
    df["target_demand_t_plus_7"] = grouped["demand_requested"].shift(-7)
    df["target_demand_t_plus_14"] = grouped["demand_requested"].shift(-14)
    df["target_demand_t_plus_28"] = grouped["demand_requested"].shift(-28)

    # Cumulative forward-looking demand over 7, 14, 28 days
    # (Sum of t+1 through t+h)
    fwd_series = grouped["demand_requested"].shift(-1)
    df["target_demand_sum_7d"] = (
        fwd_series.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.iloc[::-1].rolling(window=7, min_periods=7).sum().iloc[::-1])
    )
    df["target_demand_sum_14d"] = (
        fwd_series.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.iloc[::-1].rolling(window=14, min_periods=14).sum().iloc[::-1])
    )
    df["target_demand_sum_28d"] = (
        fwd_series.groupby([df["product_id"], df["warehouse_id"]])
        .transform(lambda s: s.iloc[::-1].rolling(window=28, min_periods=28).sum().iloc[::-1])
    )

    # -------------------------------------------------------------------------
    # 7. Metadata, Warmup & Target Availability Flags
    # -------------------------------------------------------------------------
    # First 28 days lack complete 28-day history
    df["has_sufficient_history_28d"] = (df["date"] >= (df["date"].min() + pd.Timedelta(days=28))).astype(int)

    # Targets unavailable for end-of-series dates
    df["is_target_available_t_plus_1"] = df["target_demand_t_plus_1"].notna().astype(int)
    df["is_target_available_t_plus_7"] = df["target_demand_t_plus_7"].notna().astype(int)
    df["is_target_available_t_plus_14"] = df["target_demand_t_plus_14"].notna().astype(int)
    df["is_target_available_t_plus_28"] = df["target_demand_t_plus_28"].notna().astype(int)

    # Convert date to standard ISO string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(f"Demand feature dataset constructed successfully: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df
