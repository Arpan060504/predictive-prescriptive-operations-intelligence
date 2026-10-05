"""
Sensitivity Analysis Engine for PPOI Prescriptive Optimization.

Performs parameter sweeps to evaluate how operational trade-offs respond to:
- Service Level Floors (alpha in [0.90, 0.99])
- Inbound & Transfer Freight Rates (multiplier in [0.5, 2.0])
- Sourcing Concentration Caps (gamma in [0.50, 1.00])
- Inventory Holding Cost Rates (annual rate in [0.10, 0.35])
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.optimization.optimizer import NetworkOptimizer
from src.utils.logger import get_logger

logger = get_logger("SensitivityEngine")


class SensitivityAnalyzer:
    """Executes parametric sensitivity sweeps for the Prescriptive Optimizer."""

    def __init__(
        self,
        operational_data: pd.DataFrame,
        supplier_dims: pd.DataFrame,
        warehouse_dims: pd.DataFrame,
        product_dims: pd.DataFrame,
    ):
        self.operational_data = operational_data.copy()
        self.supplier_dims = supplier_dims.copy()
        self.warehouse_dims = warehouse_dims.copy()
        self.product_dims = product_dims.copy()

    def sweep_service_level(
        self,
        targets: List[float] = [0.90, 0.92, 0.95, 0.98, 0.99]
    ) -> pd.DataFrame:
        """Evaluates cost and shortage impact across service level targets."""
        rows = []
        for target in targets:
            optimizer = NetworkOptimizer(
                operational_data=self.operational_data,
                supplier_dims=self.supplier_dims,
                warehouse_dims=self.warehouse_dims,
                product_dims=self.product_dims,
                service_level_target=target,
            )
            res = optimizer.solve(scenario_name=f"sweep_sl_{target:.2f}")
            rows.append({
                "service_level_target": target,
                "achieved_service_level": res.service_level,
                "total_cost": res.total_cost,
                "procurement_cost": res.cost_breakdown["procurement_cost"],
                "transshipment_cost": res.cost_breakdown["transshipment_cost"],
                "shortage_cost": res.cost_breakdown["shortage_cost"],
                "total_ordered": res.total_ordered,
                "total_transferred": res.total_transferred,
                "solve_time_seconds": res.solve_time_seconds,
            })
        return pd.DataFrame(rows)

    def sweep_freight_rates(
        self,
        multipliers: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ) -> pd.DataFrame:
        """Evaluates network sensitivity to freight and fuel surcharges."""
        rows = []
        for mult in multipliers:
            optimizer = NetworkOptimizer(
                operational_data=self.operational_data,
                supplier_dims=self.supplier_dims,
                warehouse_dims=self.warehouse_dims,
                product_dims=self.product_dims,
                freight_rate_base=2.50 * mult,
            )
            res = optimizer.solve(scenario_name=f"sweep_freight_{mult:.2f}")
            rows.append({
                "freight_multiplier": mult,
                "effective_base_freight": 2.50 * mult,
                "total_cost": res.total_cost,
                "transport_cost": res.cost_breakdown["transport_cost"],
                "transshipment_cost": res.cost_breakdown["transshipment_cost"],
                "total_transferred": res.total_transferred,
                "service_level": res.service_level,
                "solve_time_seconds": res.solve_time_seconds,
            })
        return pd.DataFrame(rows)

    def sweep_sourcing_cap(
        self,
        caps: List[float] = [0.50, 0.55, 0.60, 0.70, 0.85, 1.00]
    ) -> pd.DataFrame:
        """Evaluates the cost of dual-sourcing diversification."""
        rows = []
        for cap in caps:
            optimizer = NetworkOptimizer(
                operational_data=self.operational_data,
                supplier_dims=self.supplier_dims,
                warehouse_dims=self.warehouse_dims,
                product_dims=self.product_dims,
                max_supplier_allocation_pct=cap,
            )
            res = optimizer.solve(
                scenario_name=f"sweep_cap_{cap:.2f}",
                enable_dual_sourcing_cap=(cap < 1.0),
            )
            rows.append({
                "max_supplier_allocation_pct": cap,
                "total_cost": res.total_cost,
                "procurement_cost": res.cost_breakdown["procurement_cost"],
                "supplier_risk_cost": res.cost_breakdown["supplier_risk_cost"],
                "total_ordered": res.total_ordered,
                "service_level": res.service_level,
                "solve_time_seconds": res.solve_time_seconds,
            })
        return pd.DataFrame(rows)

    def run_all_sweeps(self) -> Dict[str, pd.DataFrame]:
        """Runs full suite of sensitivity sweeps."""
        logger.info("Running complete sensitivity sweep suite...")
        return {
            "service_level_sensitivity": self.sweep_service_level(),
            "freight_rate_sensitivity": self.sweep_freight_rates(),
            "sourcing_cap_sensitivity": self.sweep_sourcing_cap(),
        }
