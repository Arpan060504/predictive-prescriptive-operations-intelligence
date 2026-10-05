"""
Feature Registry Module for PPOI (Phase 6).
Provides programmatic dataclass registry and serializes feature_registry.json.
Documents feature lineage, mathematical transformations, lookback windows,
point-in-time availability rules, leakage risk scores, and downstream consumers.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FeatureMetadata:
    feature_name: str
    feature_family: str
    description: str
    transformation: str
    source_entity: str
    grain: str
    lookback_window: str
    availability_timestamp: str
    data_type: str
    leakage_risk_score: str  # "Low", "Medium", "High"
    leakage_justification: str
    downstream_consumer: str  # "Demand Forecasting", "Stockout Risk", "Supplier Risk", "Prescriptive Optimization"


REGISTRY_ENTRIES: List[FeatureMetadata] = [
    # -------------------------------------------------------------------------
    # Demand Forecasting Feature Family: Lags & Rolling
    # -------------------------------------------------------------------------
    FeatureMetadata(
        feature_name="demand_lag_1",
        feature_family="lag_demand",
        description="Realized customer demand 1 day prior to prediction cutoff",
        transformation="shift(1) on daily demand requested series",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="1 day",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Strictly lagged by 1 day; utilizes finalized previous-day order ledger.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_lag_7",
        feature_family="lag_demand",
        description="Realized customer demand exactly 1 week (7 days) prior to prediction cutoff",
        transformation="shift(7) on daily demand requested series",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="7 days",
        availability_timestamp="t-7 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Captures day-of-week seasonality strictly from 7 calendar days ago.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_lag_14",
        feature_family="lag_demand",
        description="Realized customer demand exactly 2 weeks (14 days) prior",
        transformation="shift(14) on daily demand requested series",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="14 days",
        availability_timestamp="t-14 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Lagged bi-weekly demand captures multi-week cyclicality.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_lag_28",
        feature_family="lag_demand",
        description="Realized customer demand 4 weeks (28 days) prior",
        transformation="shift(28) on daily demand requested series",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days",
        availability_timestamp="t-28 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Lagged 4-week cycle captures monthly replenishment patterns.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_rolling_mean_7",
        feature_family="rolling_demand",
        description="7-day rolling average of historical customer demand",
        transformation="shift(1).rolling(7, min_periods=1).mean()",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="7 days [t-7, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Pre-shifted by 1 day before rolling window calculation, excluding t.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_rolling_mean_28",
        feature_family="rolling_demand",
        description="28-day rolling average of historical customer demand",
        transformation="shift(1).rolling(28, min_periods=1).mean()",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Pre-shifted by 1 day; provides 4-week smoothed demand baseline.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_rolling_std_28",
        feature_family="rolling_demand",
        description="28-day rolling standard deviation of historical customer demand",
        transformation="shift(1).rolling(28, min_periods=2).std().fillna(0)",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Pre-shifted by 1 day; quantifies demand volatility without future peek.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="demand_cv_28",
        feature_family="dynamic_demand",
        description="Coefficient of variation over the past 28 days (std / mean)",
        transformation="demand_rolling_std_28 / max(demand_rolling_mean_28, 1.0)",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Derived solely from pre-shifted 28-day rolling metrics.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="zero_demand_frequency_28",
        feature_family="dynamic_demand",
        description="Proportion of days with zero demand in the past 28 days (intermittency)",
        transformation="(shift(1) == 0).rolling(28, min_periods=1).mean()",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Pre-shifted by 1 day; diagnoses Croston/intermittent demand profiles.",
        downstream_consumer="Demand Forecasting"
    ),

    # -------------------------------------------------------------------------
    # Demand Forecasting Feature Family: Calendar & Events
    # -------------------------------------------------------------------------
    FeatureMetadata(
        feature_name="day_of_week",
        feature_family="calendar",
        description="Day of week integer (0=Monday, 6=Sunday)",
        transformation="date.dt.dayofweek",
        source_entity="dim_date",
        grain="Day",
        lookback_window="0 days (deterministic)",
        availability_timestamp="Known a priori",
        data_type="int64",
        leakage_risk_score="Low",
        leakage_justification="Standard deterministic calendar feature known indefinitely in advance.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="is_weekend",
        feature_family="calendar",
        description="Binary flag indicating Saturday or Sunday",
        transformation="day_of_week.isin([5, 6]).astype(int)",
        source_entity="dim_date",
        grain="Day",
        lookback_window="0 days (deterministic)",
        availability_timestamp="Known a priori",
        data_type="int64",
        leakage_risk_score="Low",
        leakage_justification="Deterministic calendar metadata.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="event_active",
        feature_family="events",
        description="Binary indicator if a known macro disruption or promotion event is active on date t",
        transformation="date between start_date and end_date in dim_event",
        source_entity="dim_event",
        grain="Day",
        lookback_window="Event window",
        availability_timestamp="Published event calendar",
        data_type="int64",
        leakage_risk_score="Low",
        leakage_justification="Corporate promotional campaigns and planned supplier shutdowns are scheduled in advance.",
        downstream_consumer="Demand Forecasting"
    ),

    # -------------------------------------------------------------------------
    # Inventory & Stockout Risk Feature Family
    # -------------------------------------------------------------------------
    FeatureMetadata(
        feature_name="ending_inventory_lag1",
        feature_family="inventory_state",
        description="Physical ending on-hand inventory level from previous day t-1",
        transformation="shift(1) of ending_inventory",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="1 day",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Strictly lagged by 1 day; matches beginning inventory at day t start.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="safety_stock_gap",
        feature_family="buffer_health",
        description="Difference between safety stock target and available inventory (safety_stock - ending_inventory_lag1)",
        transformation="safety_stock_target - ending_inventory_lag1",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="1 day",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Uses target safety stock minus t-1 ending inventory; positive indicates deficit.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="inventory_days_of_supply",
        feature_family="buffer_health",
        description="Estimated days of inventory remaining based on recent 28-day demand burn rate",
        transformation="ending_inventory_lag1 / max(demand_mean_28, 1.0)",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Combines t-1 inventory and shifted 28-day demand burn rate.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="stockout_count_28d",
        feature_family="stockout_history",
        description="Number of stockout days experienced by this SKU-warehouse in the past 28 days",
        transformation="stockout_lag_1.rolling(28, min_periods=1).sum()",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Computed strictly over historical stockout flags from t-1 backwards.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="consecutive_stockout_days",
        feature_family="stockout_history",
        description="Length of consecutive stockout run up to day t-1",
        transformation="Cumulative run of stockout_lag_1 == 1 reset by 0",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="Variable dynamic run [t-k, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="int64",
        leakage_risk_score="Low",
        leakage_justification="Measures duration of active stockout streak ending at t-1.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="lost_sales_28d",
        feature_family="unmet_demand",
        description="Total lost sales units suffered in the past 28 days [t-28, t-1]",
        transformation="lost_sales_lag_1.rolling(28, min_periods=1).sum()",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="28 days [t-28, t-1]",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Quantifies recent customer churn and unmet demand pressure.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="warehouse_capacity_utilization_lag1",
        feature_family="warehouse_saturation",
        description="Total warehouse inventory divided by certified physical warehouse capacity at t-1",
        transformation="shift(1) of total_warehouse_inventory / warehouse_capacity",
        source_entity="fact_inventory",
        grain="Warehouse × Day",
        lookback_window="1 day",
        availability_timestamp="t-1 23:59:59",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Aggregated across all SKUs in the facility as of t-1.",
        downstream_consumer="Stockout Risk"
    ),

    # -------------------------------------------------------------------------
    # Supplier Delay Risk Feature Family
    # -------------------------------------------------------------------------
    FeatureMetadata(
        feature_name="hist_on_time_rate",
        feature_family="supplier_track_record",
        description="Historical proportion of orders delivered on or before promised delivery date",
        transformation="cumsum(on_time) / count on orders completed strictly before order_date",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Full history prior to order_date",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Filtered strictly to actual_delivery_date < order_date; orders in-transit excluded.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="hist_otif_rate",
        feature_family="supplier_track_record",
        description="Historical on-time in-full (OTIF) rate of completed deliveries",
        transformation="cumsum(otif) / count on orders completed strictly before order_date",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Full history prior to order_date",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="OTIF status of delivered purchase orders is finalized prior to new order placement.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="hist_mean_delay",
        feature_family="supplier_track_record",
        description="Historical mean delay in calendar days on completed deliveries",
        transformation="mean(delay_days) on orders completed strictly before order_date",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Full history prior to order_date",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Strict point-in-time filter prevents leaking delays of currently open orders.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="recent_delay_30d",
        feature_family="supplier_velocity",
        description="Mean delay in days for deliveries completed in the 30 days prior to order_date",
        transformation="mean(delay_days) where actual_delivery_date in [order_date - 30d, order_date)",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="30 days [order_date - 30, order_date)",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Captures immediate operational deterioration or seasonal congestion.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="delay_trend",
        feature_family="supplier_velocity",
        description="Performance velocity: recent_delay_30d minus historical lifetime mean delay",
        transformation="recent_delay_30d - hist_mean_delay",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="30 days vs lifetime",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="Difference between short-term delay and lifetime baseline indicates drift.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="supplier_capacity_pressure",
        feature_family="supplier_capacity",
        description="Ratio of total in-flight active order volume to supplier monthly capacity",
        transformation="sum(active_orders_quantity) / monthly_capacity_units",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Concurrent open orders at order_date",
        availability_timestamp="order_date 00:00:00",
        data_type="float64",
        leakage_risk_score="Low",
        leakage_justification="In-flight orders are orders placed before order_date that have not arrived by order_date.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="quantity_ordered",
        feature_family="order_specification",
        description="Quantity ordered in this specific purchase order line",
        transformation="Direct order placement attribute",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="0 days (specified at placement)",
        availability_timestamp="order_date 00:00:00",
        data_type="int64",
        leakage_risk_score="Low",
        leakage_justification="Decision variable chosen by procurement at order placement.",
        downstream_consumer="Supplier Risk"
    ),

    # -------------------------------------------------------------------------
    # Prescriptive Optimization & Target Variables
    # -------------------------------------------------------------------------
    FeatureMetadata(
        feature_name="target_demand_t_plus_1",
        feature_family="forecast_target",
        description="Forward customer demand requested 1 day ahead (t+1)",
        transformation="shift(-1) of demand_requested",
        source_entity="analytics_daily_demand",
        grain="SKU × Warehouse × Day",
        lookback_window="1 day forward (t+1)",
        availability_timestamp="Realized at t+1",
        data_type="float64",
        leakage_risk_score="High if used as feature; Verified Target Only",
        leakage_justification="Explicitly partitioned into target column vector; excluded from feature matrices.",
        downstream_consumer="Demand Forecasting"
    ),
    FeatureMetadata(
        feature_name="target_stockout_within_7d",
        feature_family="risk_target",
        description="Binary indicator whether SKU experiences at least 1 stockout day in the next 7 days [t+1, t+7]",
        transformation="shift(-1).rolling(7, min_periods=1).max() of stockout_flag",
        source_entity="fact_inventory",
        grain="SKU × Warehouse × Day",
        lookback_window="7 days forward [t+1, t+7]",
        availability_timestamp="Realized at t+7",
        data_type="int64",
        leakage_risk_score="High if used as feature; Verified Target Only",
        leakage_justification="Binary classification label for early warning stockout prevention model.",
        downstream_consumer="Stockout Risk"
    ),
    FeatureMetadata(
        feature_name="target_delay_days",
        feature_family="supplier_target",
        description="Actual delivery delay in days beyond contracted delivery date",
        transformation="actual_lead_time_days - contracted_lead_time_days",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Realized upon physical arrival",
        availability_timestamp="actual_delivery_date",
        data_type="int64",
        leakage_risk_score="High if used as feature; Verified Target Only",
        leakage_justification="Regression target for predicting supplier lead time inflation.",
        downstream_consumer="Supplier Risk"
    ),
    FeatureMetadata(
        feature_name="target_otif",
        feature_family="supplier_target",
        description="Binary indicator whether purchase order was delivered on-time and in-full (OTIF)",
        transformation="on_time_flag * in_full_flag",
        source_entity="fact_purchase_orders",
        grain="Purchase Order (PO)",
        lookback_window="Realized upon physical arrival",
        availability_timestamp="actual_delivery_date",
        data_type="int64",
        leakage_risk_score="High if used as feature; Verified Target Only",
        leakage_justification="Classification target for supplier reliability risk tiering.",
        downstream_consumer="Supplier Risk"
    )
]


def export_feature_registry(output_path: Path = Path("feature_registry.json")) -> None:
    """Exports all feature registry definitions to structured JSON."""
    data = [asdict(entry) for entry in REGISTRY_ENTRIES]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    export_feature_registry()
