"""
Scenario generation and operational stress test definitions for PPOI.

Defines 6 standardized operational supply chain scenarios:
1. Baseline: Certified demand forecasts and operating parameters.
2. Demand Surge: +20% demand increase (+35% for critical SKUs).
3. Supplier Delay Shock: High-risk suppliers suffer elevated delay probability (+0.35) and capacity contraction (-25%).
4. Transport Bottleneck: Freight rate spikes +50% and regional transit friction increases.
5. Inventory Constraint: Depleted initial inventory (-30%) simulating upstream supply shortages.
6. Combined Stress: Simultaneous demand surge (+15%), primary vendor contraction (-30%), and freight spike (+40%).
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class ScenarioDefinition:
    """Specification of an operational stress scenario."""
    scenario_id: str
    name: str
    display_name: str
    description: str
    demand_multiplier_general: float = 1.0
    demand_multiplier_critical: float = 1.0
    delay_probability_shift: float = 0.0
    freight_multiplier: float = 1.0
    supplier_capacity_multiplier: float = 1.0
    initial_inventory_multiplier: float = 1.0
    service_level_target: float = 0.95


def get_standard_scenarios() -> List[ScenarioDefinition]:
    """Returns the certified 6 standard operational scenarios."""
    return [
        ScenarioDefinition(
            scenario_id="SCEN-01",
            name="baseline",
            display_name="1. Baseline Standard Operations",
            description="Certified forward demand forecasts, standard vendor capacities, and nominal freight costs.",
            demand_multiplier_general=1.0,
            demand_multiplier_critical=1.0,
            delay_probability_shift=0.0,
            freight_multiplier=1.0,
            supplier_capacity_multiplier=1.0,
            initial_inventory_multiplier=1.0,
            service_level_target=0.95,
        ),
        ScenarioDefinition(
            scenario_id="SCEN-02",
            name="demand_surge",
            display_name="2. Demand Surge Shock",
            description="Unplanned demand spike of +20% network-wide, rising to +35% for critical components.",
            demand_multiplier_general=1.20,
            demand_multiplier_critical=1.35,
            delay_probability_shift=0.05,
            freight_multiplier=1.10,
            supplier_capacity_multiplier=1.0,
            initial_inventory_multiplier=1.0,
            service_level_target=0.95,
        ),
        ScenarioDefinition(
            scenario_id="SCEN-03",
            name="supplier_delay_shock",
            display_name="3. Supplier Delay & Disruption",
            description="Tier-2 economy vendors suffer acute delivery delays (+0.35 delay prob) and 25% capacity contraction.",
            demand_multiplier_general=1.0,
            demand_multiplier_critical=1.0,
            delay_probability_shift=0.35,
            freight_multiplier=1.05,
            supplier_capacity_multiplier=0.75,
            initial_inventory_multiplier=1.0,
            service_level_target=0.95,
        ),
        ScenarioDefinition(
            scenario_id="SCEN-04",
            name="transport_bottleneck",
            display_name="4. Transport Capacity Bottleneck",
            description="Fuel surcharges and carrier driver shortages trigger a +50% freight rate spike across regional lanes.",
            demand_multiplier_general=1.0,
            demand_multiplier_critical=1.0,
            delay_probability_shift=0.10,
            freight_multiplier=1.50,
            supplier_capacity_multiplier=1.0,
            initial_inventory_multiplier=1.0,
            service_level_target=0.95,
        ),
        ScenarioDefinition(
            scenario_id="SCEN-05",
            name="inventory_constraint",
            display_name="5. Constrained Buffer Depletion",
            description="Prior operational shortages leave initial on-hand stock depleted by 30% across all hubs.",
            demand_multiplier_general=1.0,
            demand_multiplier_critical=1.0,
            delay_probability_shift=0.0,
            freight_multiplier=1.0,
            supplier_capacity_multiplier=1.0,
            initial_inventory_multiplier=0.70,
            service_level_target=0.95,
        ),
        ScenarioDefinition(
            scenario_id="SCEN-06",
            name="combined_stress",
            display_name="6. Combined Multi-Vector Stress",
            description="Simultaneous demand surge (+15%), vendor capacity crunch (-30%), and carrier freight inflation (+40%).",
            demand_multiplier_general=1.15,
            demand_multiplier_critical=1.25,
            delay_probability_shift=0.25,
            freight_multiplier=1.40,
            supplier_capacity_multiplier=0.70,
            initial_inventory_multiplier=0.85,
            service_level_target=0.95,
        ),
    ]


def apply_scenario(
    base_operational_data: pd.DataFrame,
    base_supplier_dims: pd.DataFrame,
    base_warehouse_dims: pd.DataFrame,
    scenario: ScenarioDefinition,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies scenario modifications to copies of operational tables.
    Preserves original dataframes immutably.
    """
    op_data = base_operational_data.copy()
    sup_dims = base_supplier_dims.copy()
    wh_dims = base_warehouse_dims.copy()

    # 1. Modify demand based on SKU criticality
    if "criticality" in op_data.columns:
        is_crit = op_data["criticality"].astype(str).str.lower().isin(["critical", "high"])
        op_data.loc[is_crit, "forecast_demand_7d"] = (
            op_data.loc[is_crit, "forecast_demand_7d"] * scenario.demand_multiplier_critical
        )
        op_data.loc[~is_crit, "forecast_demand_7d"] = (
            op_data.loc[~is_crit, "forecast_demand_7d"] * scenario.demand_multiplier_general
        )
    else:
        op_data["forecast_demand_7d"] = op_data["forecast_demand_7d"] * scenario.demand_multiplier_general

    # 2. Modify starting inventory
    if scenario.initial_inventory_multiplier != 1.0:
        if "ending_inventory_lag1" in op_data.columns:
            op_data["ending_inventory_lag1"] = op_data["ending_inventory_lag1"] * scenario.initial_inventory_multiplier
        if "beginning_inventory" in op_data.columns:
            op_data["beginning_inventory"] = op_data["beginning_inventory"] * scenario.initial_inventory_multiplier

    # 3. Shift supplier delay probabilities
    if scenario.delay_probability_shift != 0.0 and "supplier_delay_probability" in op_data.columns:
        op_data["supplier_delay_probability"] = np.clip(
            op_data["supplier_delay_probability"] + scenario.delay_probability_shift, 0.0, 1.0
        )

    # 4. Modify supplier monthly capacity
    if scenario.supplier_capacity_multiplier != 1.0 and "monthly_capacity_units" in sup_dims.columns:
        sup_dims["monthly_capacity_units"] = sup_dims["monthly_capacity_units"] * scenario.supplier_capacity_multiplier

    return op_data, sup_dims, wh_dims
