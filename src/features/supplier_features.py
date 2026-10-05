"""
Supplier Delay Risk Feature Engineering Module for PPOI (Phase 6).
Constructs leakage-safe historical supplier performance metrics, capacity pressure,
and purchase order attributes for predicting delivery delay risk and OTIF reliability.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.database.queries import get_db_connection
from src.utils.logger import get_logger

logger = get_logger("SupplierFeatures")


def extract_base_purchase_orders(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Extracts all purchase order records joined with supplier dimension attributes.
    Returns 11,665 purchase orders sorted chronologically by order_date and po_id.
    """
    sql = """
    SELECT 
        po.po_id,
        po.order_date,
        po.supplier_id,
        s.supplier_name,
        s.tier AS supplier_tier,
        s.baseline_reliability,
        s.baseline_lead_time_days,
        s.lead_time_variance,
        s.monthly_capacity_units,
        s.moq_units,
        s.transport_cost_factor AS supplier_transport_cost_factor,
        po.warehouse_id,
        po.product_id,
        po.quantity_ordered,
        po.expected_lead_time_days,
        po.actual_lead_time_days,
        po.expected_delivery_date,
        po.actual_delivery_date,
        po.quantity_received,
        po.unit_cost,
        po.total_procurement_cost,
        po.transport_cost,
        po.delay_days,
        po.on_time_flag,
        po.in_full_flag,
        po.otif_flag,
        po.status
    FROM fact_purchase_orders po
    JOIN dim_supplier s ON po.supplier_id = s.supplier_id
    ORDER BY po.order_date, po.po_id;
    """
    df = pd.read_sql_query(sql, conn)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["expected_delivery_date"] = pd.to_datetime(df["expected_delivery_date"])
    df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"])
    return df


def compute_supplier_risk_features(
    conn: Optional[sqlite3.Connection] = None,
    df_override: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes leakage-safe features for each purchase order as of its order_date.
    
    Leakage Protection Policy:
    At the moment PO i is placed at order_date, the model ONLY observes:
    1. Historical orders that completed delivery STRICTLY BEFORE order_date
       (actual_delivery_date < order_date).
    2. Active in-flight orders placed before order_date that have not yet arrived
       (actual_delivery_date >= order_date), which create capacity pressure.
    3. Static supplier dimension priors (baseline reliability, capacity, tier).
    4. PO order specification (quantity_ordered, expected lead time, unit cost).
    
    Zero future delivery information (actual_lead_time, actual_delivery_date, delay_days)
    for PO i or any PO delivered on/after order_date is ever leaked into features.
    """
    if df_override is not None:
        df = df_override.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_date"] = pd.to_datetime(df["order_date"])
        if not pd.api.types.is_datetime64_any_dtype(df["actual_delivery_date"]):
            df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"])
    else:
        should_close = False
        if conn is None:
            conn = get_db_connection()
            should_close = True
        try:
            df = extract_base_purchase_orders(conn)
        finally:
            if should_close:
                conn.close()

    logger.info("Computing leakage-safe supplier risk features for %d purchase orders...", len(df))

    # Pre-allocate feature columns
    n_orders = len(df)
    hist_order_count = np.zeros(n_orders, dtype=np.int32)
    hist_order_volume = np.zeros(n_orders, dtype=np.float64)
    hist_on_time_rate = np.zeros(n_orders, dtype=np.float64)
    hist_otif_rate = np.zeros(n_orders, dtype=np.float64)
    hist_fill_rate = np.zeros(n_orders, dtype=np.float64)
    hist_mean_delay = np.zeros(n_orders, dtype=np.float64)
    hist_std_delay = np.zeros(n_orders, dtype=np.float64)
    hist_p90_delay = np.zeros(n_orders, dtype=np.float64)
    hist_p95_delay = np.zeros(n_orders, dtype=np.float64)
    recent_delay_30d = np.zeros(n_orders, dtype=np.float64)
    recent_otif_30d = np.zeros(n_orders, dtype=np.float64)
    recent_otif_90d = np.zeros(n_orders, dtype=np.float64)
    delay_trend = np.zeros(n_orders, dtype=np.float64)
    supplier_active_orders_count = np.zeros(n_orders, dtype=np.int32)
    supplier_active_volume = np.zeros(n_orders, dtype=np.float64)
    supplier_capacity_pressure = np.zeros(n_orders, dtype=np.float64)
    days_since_last_delivery = np.zeros(n_orders, dtype=np.float64)
    is_cold_start = np.zeros(n_orders, dtype=np.int32)

    # Process per supplier for maximum vectorized efficiency
    suppliers = df["supplier_id"].unique()
    order_dates_all = df["order_date"].values

    for supp_id in suppliers:
        supp_mask = (df["supplier_id"] == supp_id)
        supp_indices = np.where(supp_mask)[0]
        sub_df = df.iloc[supp_indices]
        
        baseline_rel = float(sub_df["baseline_reliability"].iloc[0])
        monthly_cap = float(sub_df["monthly_capacity_units"].iloc[0])
        # Daily capacity reference
        daily_cap = max(monthly_cap / 30.0, 1.0)

        # Completed deliveries by this supplier
        delivered_mask = sub_df["status"].isin(["RECEIVED", "DELIVERED"]) | sub_df["actual_delivery_date"].notna()
        delivered_df = sub_df[delivered_mask].sort_values("actual_delivery_date")
        
        deliv_dates = delivered_df["actual_delivery_date"].values
        deliv_delays = delivered_df["delay_days"].values.astype(np.float64)
        deliv_ontime = delivered_df["on_time_flag"].values.astype(np.float64)
        deliv_otif = delivered_df["otif_flag"].values.astype(np.float64)
        deliv_received_qty = delivered_df["quantity_received"].values.astype(np.float64)
        deliv_ordered_qty = delivered_df["quantity_ordered"].values.astype(np.float64)

        n_deliv = len(deliv_dates)
        if n_deliv > 0:
            deliv_dates_ns = deliv_dates.astype("datetime64[ns]").astype(np.int64)
            cum_delays = np.cumsum(deliv_delays)
            cum_ontime = np.cumsum(deliv_ontime)
            cum_otif = np.cumsum(deliv_otif)
            cum_received_qty = np.cumsum(deliv_received_qty)
            cum_ordered_qty = np.cumsum(deliv_ordered_qty)
        else:
            deliv_dates_ns = np.array([], dtype=np.int64)

        # In-flight tracking: all orders by this supplier
        all_order_dates_ns = sub_df["order_date"].values.astype("datetime64[ns]").astype(np.int64)
        all_deliv_dates_ns = sub_df["actual_delivery_date"].values.astype("datetime64[ns]").astype(np.int64)
        all_order_qtys = sub_df["quantity_ordered"].values.astype(np.float64)

        ns_per_day = 86_400_000_000_000  # nanoseconds per day
        ns_30d = 30 * ns_per_day
        ns_90d = 90 * ns_per_day

        for loc_idx, global_idx in enumerate(supp_indices):
            t_order = all_order_dates_ns[loc_idx]

            # 1. Historical completed orders strictly before t_order
            # Using binary search: index of first delivery >= t_order
            k = np.searchsorted(deliv_dates_ns, t_order, side="left")

            if k == 0:
                # Cold start period
                is_cold_start[global_idx] = 1
                hist_order_count[global_idx] = 0
                hist_order_volume[global_idx] = 0.0
                hist_on_time_rate[global_idx] = baseline_rel
                hist_otif_rate[global_idx] = baseline_rel
                hist_fill_rate[global_idx] = 1.0
                hist_mean_delay[global_idx] = 0.0
                hist_std_delay[global_idx] = 0.0
                hist_p90_delay[global_idx] = 0.0
                hist_p95_delay[global_idx] = 0.0
                recent_delay_30d[global_idx] = 0.0
                recent_otif_30d[global_idx] = baseline_rel
                recent_otif_90d[global_idx] = baseline_rel
                delay_trend[global_idx] = 0.0
                days_since_last_delivery[global_idx] = 999.0
            else:
                is_cold_start[global_idx] = 0
                hist_order_count[global_idx] = k
                hist_order_volume[global_idx] = cum_received_qty[k - 1]
                hist_on_time_rate[global_idx] = cum_ontime[k - 1] / k
                hist_otif_rate[global_idx] = cum_otif[k - 1] / k
                hist_fill_rate[global_idx] = cum_received_qty[k - 1] / max(cum_ordered_qty[k - 1], 1.0)
                mean_d = cum_delays[k - 1] / k
                hist_mean_delay[global_idx] = mean_d

                # Std and percentiles on observed slice
                slice_delays = deliv_delays[:k]
                hist_std_delay[global_idx] = np.std(slice_delays, ddof=1) if k > 1 else 0.0
                hist_p90_delay[global_idx] = np.percentile(slice_delays, 90)
                hist_p95_delay[global_idx] = np.percentile(slice_delays, 95)

                # Days since last delivery
                last_deliv_time = deliv_dates_ns[k - 1]
                days_since_last_delivery[global_idx] = (t_order - last_deliv_time) / ns_per_day

                # Recent 30-day window [t_order - 30d, t_order)
                j_30 = np.searchsorted(deliv_dates_ns, t_order - ns_30d, side="left")
                n_30 = k - j_30
                if n_30 > 0:
                    sum_d_30 = cum_delays[k - 1] - (cum_delays[j_30 - 1] if j_30 > 0 else 0.0)
                    sum_otif_30 = cum_otif[k - 1] - (cum_otif[j_30 - 1] if j_30 > 0 else 0.0)
                    rec_d30 = sum_d_30 / n_30
                    recent_delay_30d[global_idx] = rec_d30
                    recent_otif_30d[global_idx] = sum_otif_30 / n_30
                else:
                    recent_delay_30d[global_idx] = mean_d
                    recent_otif_30d[global_idx] = hist_otif_rate[global_idx]
                
                delay_trend[global_idx] = recent_delay_30d[global_idx] - mean_d

                # Recent 90-day window [t_order - 90d, t_order)
                j_90 = np.searchsorted(deliv_dates_ns, t_order - ns_90d, side="left")
                n_90 = k - j_90
                if n_90 > 0:
                    sum_otif_90 = cum_otif[k - 1] - (cum_otif[j_90 - 1] if j_90 > 0 else 0.0)
                    recent_otif_90d[global_idx] = sum_otif_90 / n_90
                else:
                    recent_otif_90d[global_idx] = hist_otif_rate[global_idx]

            # 2. In-flight active orders: placed before t_order and delivered >= t_order
            active_mask = (all_order_dates_ns < t_order) & (all_deliv_dates_ns >= t_order)
            active_count = np.sum(active_mask)
            active_vol = np.sum(all_order_qtys[active_mask]) if active_count > 0 else 0.0

            supplier_active_orders_count[global_idx] = active_count
            supplier_active_volume[global_idx] = active_vol
            supplier_capacity_pressure[global_idx] = active_vol / max(monthly_cap, 1.0)

    # Assign computed columns to DataFrame
    df["hist_order_count"] = hist_order_count
    df["hist_order_volume"] = hist_order_volume
    df["hist_on_time_rate"] = hist_on_time_rate
    df["hist_otif_rate"] = hist_otif_rate
    df["hist_fill_rate"] = hist_fill_rate
    df["hist_mean_delay"] = hist_mean_delay
    df["hist_std_delay"] = hist_std_delay
    df["hist_p90_delay"] = hist_p90_delay
    df["hist_p95_delay"] = hist_p95_delay
    df["recent_delay_30d"] = recent_delay_30d
    df["recent_otif_30d"] = recent_otif_30d
    df["recent_otif_90d"] = recent_otif_90d
    df["delay_trend"] = delay_trend
    df["supplier_active_orders_count"] = supplier_active_orders_count
    df["supplier_active_volume"] = supplier_active_volume
    df["supplier_capacity_pressure"] = supplier_capacity_pressure
    df["days_since_last_delivery"] = days_since_last_delivery
    df["is_cold_start"] = is_cold_start

    # Order and calendar context available at order placement
    df["order_day_of_week"] = df["order_date"].dt.dayofweek
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)
    df["order_is_month_end"] = df["order_date"].dt.is_month_end.astype(int)

    # Historical SKU unit cost volatility
    df["hist_sku_unit_cost_mean"] = df.groupby(["supplier_id", "product_id"])["unit_cost"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["unit_cost"])
    df["hist_sku_unit_cost_std"] = df.groupby(["supplier_id", "product_id"])["unit_cost"].transform(
        lambda s: s.shift(1).expanding().std()
    ).fillna(0.0)

    # Targets for ML prediction
    df["target_delay_days"] = df["delay_days"]
    df["target_delay_gt_2d"] = (df["delay_days"] > 2).astype(int)
    df["target_delay_gt_5d"] = (df["delay_days"] > 5).astype(int)
    df["target_on_time"] = df["on_time_flag"]
    df["target_otif"] = df["otif_flag"]
    df["target_actual_lead_time_days"] = df["actual_lead_time_days"]

    logger.info("Supplier risk feature engineering complete: %d rows × %d columns.", df.shape[0], df.shape[1])
    return df
