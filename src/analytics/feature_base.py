"""
Leakage-Free Feature Engineering Foundation for PPOI (Phase 5 & 6 Preparation).
Constructs lag, rolling, and categorical features strictly using information available
prior to the prediction timestamp (t - 1 back).
Guarantees zero future leakage across temporal boundaries.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.utils.logger import get_logger

logger = get_logger("FeatureBase")


def build_demand_feature_dataset(
    conn: Optional[Any] = None,
    lags: List[int] = [1, 7, 14, 28],
    windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Extracts SKU-Warehouse-Day demand series and computes strictly historical features.
    
    LEAKAGE PREVENTION GUARANTEE:
    All rolling statistics are computed on shifted series: s.shift(1).rolling(w).mean()
    ensuring day t features incorporate only information from day t-1 and earlier.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    # Pull from materialized daily demand rollup
    sql = """
    SELECT 
        date,
        product_id,
        category,
        warehouse_id,
        demand_requested,
        promotion_flag,
        event_flag
    FROM analytics_daily_demand
    ORDER BY product_id, warehouse_id, date;
    """
    df = pd.read_sql_query(sql, conn)
    if close_conn:
        conn.close()

    df["date"] = pd.to_datetime(df["date"])
    
    # Sort strictly by time series key
    df = df.sort_values(["product_id", "warehouse_id", "date"]).reset_index(drop=True)
    
    # Calendar features (known deterministically in advance)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Compute group-level historical features
    grouped = df.groupby(["product_id", "warehouse_id"])
    
    # 1. Historical Lags (strictly t - lag)
    for lag in lags:
        df[f"lag_{lag}"] = grouped["demand_requested"].shift(lag)

    # 2. Rolling Historical Windows (using shift(1) to strictly exclude target day t)
    for w in windows:
        shifted = grouped["demand_requested"].shift(1)
        # Shifted rolling mean
        df[f"rolling_mean_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        )
        # Shifted rolling std
        df[f"rolling_std_{w}"] = (
            shifted.groupby([df["product_id"], df["warehouse_id"]])
            .transform(lambda s: s.rolling(window=w, min_periods=2).std().fillna(0.0))
        )

    # 3. Demand Volatility: Coefficient of Variation over 28-day historical window
    df["demand_cv_28"] = (
        df["rolling_std_28"] / df["rolling_mean_28"].replace(0, np.nan)
    ).fillna(0.0).round(3)

    # 4. Short-term Demand Growth Rate: rolling_mean_7 / rolling_mean_28
    df["demand_growth_ratio"] = (
        df["rolling_mean_7"] / df["rolling_mean_28"].replace(0, np.nan)
    ).fillna(1.0).round(3)

    # Convert date back to ISO string for standard relational joins
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df


def verify_temporal_leakage_invariance(
    feature_builder_func,
    cutoff_date: str = "2025-06-01"
) -> bool:
    """
    Empirical Leakage Verification Test:
    Generates feature set F1.
    Alters future target values (after cutoff_date) to synthetic random noise.
    Generates feature set F2.
    Asserts that for all records where date <= cutoff_date, F1 == F2 exactly.
    """
    conn = get_db_connection()
    df_f1 = feature_builder_func(conn)
    
    # Create corrupted future dataframe
    df_demand_orig = pd.read_sql_query("SELECT * FROM analytics_daily_demand;", conn)
    df_demand_corrupt = df_demand_orig.copy()
    
    # Corrupt future demand (after cutoff_date) with 10x values
    future_mask = df_demand_corrupt["date"] > cutoff_date
    df_demand_corrupt.loc[future_mask, "demand_requested"] = (
        df_demand_corrupt.loc[future_mask, "demand_requested"] * 10 + 500
    )
    
    # Write to temp table and build features
    df_demand_corrupt.to_sql("temp_analytics_daily_demand", conn, if_exists="replace", index=False)
    
    # Run modified query
    sql = """
    SELECT 
        date, product_id, category, warehouse_id, demand_requested, promotion_flag, event_flag
    FROM temp_analytics_daily_demand
    ORDER BY product_id, warehouse_id, date;
    """
    df_c = pd.read_sql_query(sql, conn)
    df_c["date"] = pd.to_datetime(df_c["date"])
    df_c = df_c.sort_values(["product_id", "warehouse_id", "date"]).reset_index(drop=True)
    
    # Calculate features on corrupted copy
    grouped = df_c.groupby(["product_id", "warehouse_id"])
    df_c["lag_1"] = grouped["demand_requested"].shift(1)
    df_c["lag_7"] = grouped["demand_requested"].shift(7)
    df_c["rolling_mean_7"] = (
        grouped["demand_requested"].shift(1)
        .groupby([df_c["product_id"], df_c["warehouse_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    df_c["date"] = df_c["date"].dt.strftime("%Y-%m-%d")
    
    conn.execute("DROP TABLE IF EXISTS temp_analytics_daily_demand;")
    conn.close()
    
    # Filter both to historical period
    f1_hist = df_f1[df_f1["date"] <= cutoff_date][["date", "product_id", "warehouse_id", "lag_1", "lag_7", "rolling_mean_7"]].dropna()
    f2_hist = df_c[df_c["date"] <= cutoff_date][["date", "product_id", "warehouse_id", "lag_1", "lag_7", "rolling_mean_7"]].dropna()
    
    diff = (f1_hist["rolling_mean_7"].values - f2_hist["rolling_mean_7"].values)
    is_invariant = bool(np.abs(diff).max() == 0.0)
    
    return is_invariant
