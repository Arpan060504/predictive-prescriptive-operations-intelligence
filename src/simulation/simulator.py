"""
Batch Scenario Simulator and Policy Benchmarking Engine for PPOI.

Executes comparative multi-scenario simulations across both the Constrained Optimizer
and the Deterministic Baseline Policy, capturing cost trade-offs, service levels,
and operational vulnerability profiles.
"""

from typing import Dict, List, Any, Optional
import time
import pandas as pd

from src.optimization.optimizer import NetworkOptimizer, OptimizationResult
from src.optimization.baseline import BaselinePolicyEngine, BaselineResult
from src.simulation.scenarios import get_standard_scenarios, apply_scenario, ScenarioDefinition
from src.utils.logger import get_logger

logger = get_logger("ScenarioSimulator")


class ScenarioSimulator:
    """Orchestrates multi-scenario comparative simulations."""

    def __init__(
        self,
        operational_data: pd.DataFrame,
        supplier_dims: pd.DataFrame,
        warehouse_dims: pd.DataFrame,
        product_dims: pd.DataFrame,
        service_level_target: float = 0.95,
        max_supplier_allocation_pct: float = 0.60,
        freight_rate_base: float = 2.50,
    ):
        self.operational_data = operational_data.copy()
        self.supplier_dims = supplier_dims.copy()
        self.warehouse_dims = warehouse_dims.copy()
        self.product_dims = product_dims.copy()

        self.service_level_target = service_level_target
        self.max_supplier_allocation_pct = max_supplier_allocation_pct
        self.freight_rate_base = freight_rate_base

        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all(self, scenarios: Optional[List[ScenarioDefinition]] = None) -> pd.DataFrame:
        """
        Runs both the Optimizer and Baseline heuristic across all specified scenarios.
        
        Returns:
            Summary comparison DataFrame.
        """
        if scenarios is None:
            scenarios = get_standard_scenarios()

        summary_rows = []
        logger.info(f"Initiating simulation run across {len(scenarios)} operational scenarios...")

        for sc in scenarios:
            logger.info(f"Simulating Scenario '{sc.name}' ({sc.display_name})...")
            # Apply scenario modifiers
            op_sc, sup_sc, wh_sc = apply_scenario(
                self.operational_data, self.supplier_dims, self.warehouse_dims, sc
            )

            # 1. Run Baseline Heuristic
            baseline_engine = BaselinePolicyEngine(
                operational_data=op_sc,
                supplier_dims=sup_sc,
                warehouse_dims=wh_sc,
                product_dims=self.product_dims,
                freight_rate_base=self.freight_rate_base,
            )
            base_res = baseline_engine.evaluate(
                scenario_name=sc.name,
                freight_multiplier=sc.freight_multiplier,
            )

            # 2. Run Prescriptive Optimizer
            optimizer = NetworkOptimizer(
                operational_data=op_sc,
                supplier_dims=sup_sc,
                warehouse_dims=wh_sc,
                product_dims=self.product_dims,
                service_level_target=sc.service_level_target,
                max_supplier_allocation_pct=self.max_supplier_allocation_pct,
                freight_rate_base=self.freight_rate_base,
            )
            opt_res = optimizer.solve(
                scenario_name=sc.name,
                freight_multiplier=sc.freight_multiplier,
                enable_transfers=True,
                enable_dual_sourcing_cap=True,
            )

            cost_diff = base_res.total_cost - opt_res.total_cost
            cost_reduction_pct = (cost_diff / base_res.total_cost * 100.0) if base_res.total_cost > 0 else 0.0

            base_shortage = float(base_res.shortages["shortage_quantity"].sum()) if not base_res.shortages.empty else 0.0
            opt_shortage = float(opt_res.shortages["shortage_quantity"].sum()) if not opt_res.shortages.empty else 0.0

            summary_rows.append({
                "scenario_id": sc.scenario_id,
                "scenario_name": sc.name,
                "display_name": sc.display_name,
                "baseline_total_cost": base_res.total_cost,
                "optimized_total_cost": opt_res.total_cost,
                "cost_difference": round(cost_diff, 2),
                "cost_reduction_pct": round(cost_reduction_pct, 2),
                "baseline_service_level": base_res.service_level,
                "optimized_service_level": opt_res.service_level,
                "baseline_shortage_units": round(base_shortage, 2),
                "optimized_shortage_units": round(opt_shortage, 2),
                "lateral_transfers_units": opt_res.total_transferred,
                "total_demand_units": opt_res.total_demand,
                "solve_time_seconds": opt_res.solve_time_seconds,
            })

            self.results[sc.name] = {
                "scenario_definition": sc,
                "baseline_result": base_res,
                "optimization_result": opt_res,
            }

        df_summary = pd.DataFrame(summary_rows)
        logger.info("Simulation run completed successfully.")
        return df_summary


def run_scenario_generation() -> pd.DataFrame:
    """CLI orchestrator for scenario simulation."""
    import sqlite3
    from pathlib import Path
    from src.utils.config import get_resolved_path

    logger.info("Starting scenario simulation run from CLI...")
    from src.database.connection import get_connection
    conn = get_connection(readonly=True)
    p_dims = pd.read_sql("SELECT * FROM dim_product", conn)
    s_dims = pd.read_sql("SELECT * FROM dim_supplier", conn)
    w_dims = pd.read_sql("SELECT * FROM dim_warehouse", conn)
    conn.close()

    op_path = Path("data/processed/risk/operational_risk_priorities.parquet")
    if not op_path.exists():
        raise FileNotFoundError(f"Missing certified risk priorities file at {op_path}")

    op_df = pd.read_parquet(op_path)
    latest_date = op_df["date"].max()
    op_latest = op_df[op_df["date"] == latest_date].copy()

    simulator = ScenarioSimulator(
        operational_data=op_latest,
        supplier_dims=s_dims,
        warehouse_dims=w_dims,
        product_dims=p_dims,
    )

    summary_df = simulator.run_all()
    print("\n" + "=" * 90)
    print("PRESCRIPTIVE SCENARIO SIMULATION SUMMARY (BASELINE VS OPTIMIZED)")
    print("=" * 90)
    for _, r in summary_df.iterrows():
        print(f"Scenario: {r['display_name']:<35}")
        print(f"  Baseline Cost    : ${r['baseline_total_cost']:>12,.2f}  |  Fill Rate: {r['baseline_service_level']*100:.1f}%")
        print(f"  Optimized Cost   : ${r['optimized_total_cost']:>12,.2f}  |  Fill Rate: {r['optimized_service_level']*100:.1f}%")
        print(f"  Modeled Diff     : -${r['cost_difference']:>11,.2f} ({r['cost_reduction_pct']:.1f}% lower) | Transfers: {r['lateral_transfers_units']:>6,.0f} units")
        print("-" * 90)
    print("=" * 90 + "\n")

    return summary_df
