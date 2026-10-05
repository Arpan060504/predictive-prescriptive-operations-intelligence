"""
Authoritative KPI Definitions & Metrics Registry for PPOI.
Defines every operational and financial metric with exact mathematical formulas,
source tables, SQL implementations, measurement grains, units, and operational boundaries.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class KPIDefinition:
    kpi_id: str
    name: str
    business_definition: str
    mathematical_formula: str
    source_tables: List[str]
    sql_implementation: str
    grain: str
    unit: str
    known_limitations: str


KPI_REGISTRY: Dict[str, KPIDefinition] = {
    "KPI-001": KPIDefinition(
        kpi_id="KPI-001",
        name="Unit Fill Service Level",
        business_definition="The volumetric percentage of total requested customer demand fulfilled from available warehouse inventory.",
        mathematical_formula="SUM(demand_fulfilled) / SUM(demand_requested) * 100",
        source_tables=["fact_inventory", "fact_demand"],
        sql_implementation="ROUND(CAST(SUM(i.demand_fulfilled) AS REAL) / NULLIF(SUM(i.demand_requested), 0) * 100, 2)",
        grain="SKU × Warehouse × Period (or aggregated Network-wide)",
        unit="Percentage (%)",
        known_limitations="High-volume, low-criticality SKUs can mask stockouts on low-volume, high-criticality spare parts when aggregated globally."
    ),
    "KPI-002": KPIDefinition(
        kpi_id="KPI-002",
        name="Stockout Frequency Rate",
        business_definition="The proportion of operational SKU-Warehouse-Days where customer demand could not be completely fulfilled due to inventory exhaustion.",
        mathematical_formula="COUNT(stockout_flag = 1) / COUNT(*) * 100",
        source_tables=["fact_inventory"],
        sql_implementation="ROUND(CAST(SUM(stockout_flag) AS REAL) / COUNT(*) * 100, 2)",
        grain="SKU × Warehouse × Day",
        unit="Percentage (%)",
        known_limitations="Treats a 1-unit shortage identically to a 500-unit shortage; should be evaluated alongside Lost Sales Volume."
    ),
    "KPI-003": KPIDefinition(
        kpi_id="KPI-003",
        name="Lost Sales Rate",
        business_definition="The proportion of customer demand permanently lost to competitors when stockouts occur (30% unbackordered shortage under commercial policy).",
        mathematical_formula="SUM(lost_sales_quantity) / SUM(demand_requested) * 100",
        source_tables=["fact_inventory"],
        sql_implementation="ROUND(CAST(SUM(lost_sales_quantity) AS REAL) / NULLIF(SUM(demand_requested), 0) * 100, 2)",
        grain="SKU × Warehouse × Period",
        unit="Percentage (%)",
        known_limitations="Depends on the assumed commercial customer patience ratio (70% backorder vs. 30% lost sales)."
    ),
    "KPI-004": KPIDefinition(
        kpi_id="KPI-004",
        name="Annualized Inventory Turnover",
        business_definition="The rate at which warehouse inventory is sold and replenished over a 365.25-day annualized operating cycle.",
        mathematical_formula="(SUM(demand_fulfilled) / AVG(ending_inventory)) * (365.25 / Observation_Days)",
        source_tables=["fact_inventory"],
        sql_implementation="ROUND((CAST(SUM(demand_fulfilled) AS REAL) / NULLIF(AVG(ending_inventory), 0)) * (365.25 / 731.0), 2)",
        grain="SKU × Warehouse × Annual Horizon",
        unit="Ratio (Turns / Year)",
        known_limitations="Can become volatile or infinite if average ending inventory approaches zero during chronic stockout periods."
    ),
    "KPI-005": KPIDefinition(
        kpi_id="KPI-005",
        name="Days of Inventory Coverage (DOIC)",
        business_definition="The estimated number of operating days current ending inventory will sustain based on recent average daily fulfilled demand.",
        mathematical_formula="ending_inventory / (recent_period_fulfilled_units / days_in_period)",
        source_tables=["fact_inventory"],
        sql_implementation="ROUND(ending_inventory / NULLIF(rolling_mean_demand, 0), 1)",
        grain="SKU × Warehouse × Day",
        unit="Operating Days",
        known_limitations="Sensitive to sudden demand spikes or seasonality shifts that render historical daily burn rates unrepresentative."
    ),
    "KPI-006": KPIDefinition(
        kpi_id="KPI-006",
        name="Safety Stock Coverage Ratio",
        business_definition="The ratio of physical ending inventory on hand relative to the recommended statistical safety stock buffer target.",
        mathematical_formula="ending_inventory / safety_stock_target",
        source_tables=["fact_inventory"],
        sql_implementation="ROUND(CAST(ending_inventory AS REAL) / NULLIF(safety_stock_target, 0), 2)",
        grain="SKU × Warehouse × Day",
        unit="Ratio (< 1.0 indicates safety stock breach)",
        known_limitations="Safety stock target assumes normally distributed lead-time demand, which may underestimate risk during multimodal disruption events."
    ),
    "KPI-007": KPIDefinition(
        kpi_id="KPI-007",
        name="Supplier On-Time Delivery Rate",
        business_definition="The percentage of replenishment purchase orders delivered on or before the contractual expected delivery date.",
        mathematical_formula="SUM(on_time_flag) / COUNT(po_id) * 100",
        source_tables=["fact_purchase_orders"],
        sql_implementation="ROUND(AVG(on_time_flag) * 100, 2)",
        grain="Supplier × Period",
        unit="Percentage (%)",
        known_limitations="Does not measure whether the delivered quantity was complete (in-full); must be combined with OTIF."
    ),
    "KPI-008": KPIDefinition(
        kpi_id="KPI-008",
        name="Supplier OTIF (On-Time In-Full) Rate",
        business_definition="The percentage of purchase orders that met both conditions: delivered on or before expected delivery date AND delivered 100% of ordered quantity.",
        mathematical_formula="SUM(otif_flag) / COUNT(po_id) * 100",
        source_tables=["fact_purchase_orders"],
        sql_implementation="ROUND(AVG(otif_flag) * 100, 2)",
        grain="Supplier × Period",
        unit="Percentage (%)",
        known_limitations="Strict binary metric: an order arriving 1 hour late or with 99.5% quantity receives a score of 0."
    ),
    "KPI-009": KPIDefinition(
        kpi_id="KPI-009",
        name="Average Supplier Lead-Time Delay",
        business_definition="The mean number of calendar days shipment arrivals were delayed beyond contractual expected delivery dates.",
        mathematical_formula="AVG(max(0, actual_delivery_date - expected_delivery_date))",
        source_tables=["fact_purchase_orders"],
        sql_implementation="ROUND(AVG(delay_days), 2)",
        grain="Supplier × Period",
        unit="Calendar Days",
        known_limitations="Mean is pulled upward by extreme outlier events; should be paired with Median and P90/P95."
    ),
    "KPI-010": KPIDefinition(
        kpi_id="KPI-010",
        name="P90 / P95 Lead-Time Delay",
        business_definition="The 90th and 95th percentile upper-tail delays experienced on inbound shipments.",
        mathematical_formula="Quantile(delay_days, 0.90) and Quantile(delay_days, 0.95)",
        source_tables=["fact_purchase_orders"],
        sql_implementation="Calculated via ordered rank interpolation over PO delay distributions",
        grain="Supplier × Annual/Quarterly Horizon",
        unit="Calendar Days",
        known_limitations="Requires sufficient order sample size (>= 30 POs) per supplier to be statistically stable."
    ),
    "KPI-011": KPIDefinition(
        kpi_id="KPI-011",
        name="Inventory Holding Cost",
        business_definition="The carrying cost of holding physical stock in warehouses, calculated at a 20% annualized holding rate.",
        mathematical_formula="SUM(ending_inventory × unit_cost × (annual_holding_rate / 365.25))",
        source_tables=["fact_inventory", "dim_product"],
        sql_implementation="ROUND(SUM(holding_cost), 2)",
        grain="SKU × Warehouse × Day (aggregated to Period)",
        unit="Currency ($ USD)",
        known_limitations="Assumes constant 20% cost of capital; does not factor seasonal changes in commercial borrowing interest rates."
    ),
    "KPI-012": KPIDefinition(
        kpi_id="KPI-012",
        name="Stockout Penalty Cost",
        business_definition="The financial penalty and lost contribution margin incurred on lost sales, valued at 1.5x product unit selling price.",
        mathematical_formula="SUM(lost_sales_quantity × selling_price × 1.50)",
        source_tables=["fact_inventory", "dim_product"],
        sql_implementation="ROUND(SUM(stockout_cost), 2)",
        grain="SKU × Warehouse × Day (aggregated to Period)",
        unit="Currency ($ USD)",
        known_limitations="Applies strictly to unrecoverable lost sales (30% split); does not charge holding penalties to deferred backorders."
    ),
    "KPI-013": KPIDefinition(
        kpi_id="KPI-013",
        name="Total Procurement Cost",
        business_definition="Direct vendor acquisition expenditures on delivered purchase orders.",
        mathematical_formula="SUM(quantity_received × unit_cost)",
        source_tables=["fact_purchase_orders"],
        sql_implementation="ROUND(SUM(total_procurement_cost), 2)",
        grain="Supplier × SKU × Period",
        unit="Currency ($ USD)",
        known_limitations="Excludes customs duties or import tariffs unless baked into the supplier base unit cost."
    ),
    "KPI-014": KPIDefinition(
        kpi_id="KPI-014",
        name="Transportation Freight Cost",
        business_definition="Inbound logistics freight shipping charges between supplier origin facilities and destination distribution warehouses.",
        mathematical_formula="SUM(order_quantity × $2.50 × wh_factor × sup_factor × event_multiplier)",
        source_tables=["fact_purchase_orders", "fact_transport"],
        sql_implementation="ROUND(SUM(transport_cost), 2)",
        grain="Transport Shipment × Purchase Order",
        unit="Currency ($ USD)",
        known_limitations="Freight costs are fixed at contract baseline modulated by macro logistics disruption surcharges."
    ),
    "KPI-015": KPIDefinition(
        kpi_id="KPI-015",
        name="Total Operational Cost",
        business_definition="Comprehensive supply chain operating cost across procurement, inventory holding, stockout penalties, and freight transport.",
        mathematical_formula="Procurement Cost + Holding Cost + Stockout Penalty Cost + Transportation Cost",
        source_tables=["fact_purchase_orders", "fact_inventory", "fact_transport"],
        sql_implementation="SUM(procurement) + SUM(holding) + SUM(stockout) + SUM(transport)",
        grain="Network × Period",
        unit="Currency ($ USD)",
        known_limitations="Must be calculated via pre-aggregated CTEs to prevent Cartesian measure inflation."
    ),
    "KPI-016": KPIDefinition(
        kpi_id="KPI-016",
        name="Warehouse Capacity Utilization %",
        business_definition="The volumetric percentage of total physical warehouse storage capacity occupied by ending inventory across all stored SKUs.",
        mathematical_formula="(SUM_SKU(ending_inventory) / warehouse_capacity) * 100",
        source_tables=["fact_inventory", "dim_warehouse"],
        sql_implementation="ROUND((CAST(SUM(ending_inventory) AS REAL) / MAX(warehouse_capacity)) * 100, 2)",
        grain="Warehouse × Day",
        unit="Percentage (%)",
        known_limitations="Assumes homogenous unit volume footprint across categories; does not distinguish high-density pallets from bulky boxes."
    ),
    "KPI-017": KPIDefinition(
        kpi_id="KPI-017",
        name="Demand Volatility (Coefficient of Variation)",
        business_definition="The standard deviation of daily demand normalized by mean daily demand, quantifying demand volatility regime.",
        mathematical_formula="std_dev(daily_demand) / mean(daily_demand)",
        source_tables=["fact_demand"],
        sql_implementation="SQRT(AVG(demand*demand) - AVG(demand)*AVG(demand)) / NULLIF(AVG(demand), 0)",
        grain="SKU × Warehouse",
        unit="Dimensionless Ratio (< 0.5 Smooth, 0.5 - 1.0 Variable, > 1.0 Lumpy/Erratic)",
        known_limitations="Can yield skewed volatility metrics if computed over very short observation windows (< 30 days)."
    )
}


def get_kpi_registry() -> Dict[str, KPIDefinition]:
    """Returns the complete authoritative dictionary of operational KPIs."""
    return KPI_REGISTRY
