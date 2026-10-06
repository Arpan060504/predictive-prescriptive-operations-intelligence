"""
Prescriptive Optimization Orchestrator for PPOI Platform.

Integrates data ingestion, model solving, scenario simulation, sensitivity analysis,
explainability generation, artifact serialization, and SQLite persistence.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import pandas as pd

from src.optimization.optimizer import NetworkOptimizer, OptimizationResult
from src.optimization.baseline import BaselinePolicyEngine, BaselineResult
from src.simulation.scenarios import get_standard_scenarios
from src.simulation.simulator import ScenarioSimulator
from src.decision_engine.policy_comparison import compare_policies
from src.decision_engine.sensitivity import SensitivityAnalyzer
from src.decision_engine.explanations import DecisionExplanationEngine
from src.optimization.persistence import (
    init_optimization_tables,
    save_scenario_definitions,
    record_optimization_run,
    record_policy_comparison,
    record_explanations,
)
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("PrescriptiveEngine")


def run_prescriptive_optimization() -> Dict[str, Any]:
    """
    Executes end-to-end Prescriptive Operations Optimization:
    1. Loads certified test operational data and master dimensions.
    2. Solves baseline vs optimized network models.
    3. Simulates 6 operational stress scenarios.
    4. Executes sensitivity sweeps.
    5. Derives deterministic decision explanations.
    6. Persists data to SQLite tables and parquet artifacts.
    """
    logger.info("Initializing PPOI Prescriptive Optimization Pipeline...")
    config = load_config()

    # Paths
    from src.database.connection import get_database_path, get_connection
    db_path = get_database_path()
    op_path = get_resolved_path("data/processed/risk/operational_risk_priorities.parquet")
    output_dir = get_resolved_path("data/processed/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not op_path.exists():
        raise FileNotFoundError(f"Missing certified operational risk data at {op_path}")

    # Load Database Master Tables
    conn = get_connection(db_path)
    p_dims = pd.read_sql("SELECT * FROM dim_product", conn)
    s_dims = pd.read_sql("SELECT * FROM dim_supplier", conn)
    w_dims = pd.read_sql("SELECT * FROM dim_warehouse", conn)

    # Initialize optimization tables
    init_optimization_tables(conn)
    standard_scenarios = get_standard_scenarios()
    save_scenario_definitions(conn, standard_scenarios)

    # Load Operational Priorities (Decision Date = Latest Test Date)
    op_df = pd.read_parquet(op_path)
    latest_date = op_df["date"].max()
    op_latest = op_df[op_df["date"] == latest_date].copy()
    logger.info(f"Loaded {len(op_latest)} operational positions at decision date {latest_date}")

    opt_cfg = config.get("optimization", {})
    service_level_target = float(opt_cfg.get("default_service_level_target", 0.95))
    max_supplier_allocation_pct = float(opt_cfg.get("max_supplier_allocation_pct", 0.60))

    # 1. Solve Baseline Heuristic
    baseline_engine = BaselinePolicyEngine(
        operational_data=op_latest,
        supplier_dims=s_dims,
        warehouse_dims=w_dims,
        product_dims=p_dims,
    )
    base_res = baseline_engine.evaluate(scenario_name="baseline")

    # 2. Solve Prescriptive Optimizer (Baseline Scenario)
    optimizer = NetworkOptimizer(
        operational_data=op_latest,
        supplier_dims=s_dims,
        warehouse_dims=w_dims,
        product_dims=p_dims,
        service_level_target=service_level_target,
        max_supplier_allocation_pct=max_supplier_allocation_pct,
    )
    opt_res = optimizer.solve(scenario_name="baseline")

    # Record runs to database
    run_id_opt = f"OPT-BASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    record_optimization_run(conn, opt_res, run_id_opt)

    # 3. Policy Comparison
    comparison = compare_policies(opt_res, base_res)

    # 4. Multi-Scenario Simulation
    simulator = ScenarioSimulator(
        operational_data=op_latest,
        supplier_dims=s_dims,
        warehouse_dims=w_dims,
        product_dims=p_dims,
        service_level_target=service_level_target,
        max_supplier_allocation_pct=max_supplier_allocation_pct,
    )
    scenario_summary_df = simulator.run_all(standard_scenarios)
    record_policy_comparison(conn, scenario_summary_df)

    # 5. Sensitivity Sweeps
    sensitivity_engine = SensitivityAnalyzer(
        operational_data=op_latest,
        supplier_dims=s_dims,
        warehouse_dims=w_dims,
        product_dims=p_dims,
    )
    sensitivity_results = sensitivity_engine.run_all_sweeps()

    # 6. Deterministic Explanations
    explanation_engine = DecisionExplanationEngine(p_dims, s_dims, w_dims)
    xfer_exps = explanation_engine.explain_transfers(opt_res, top_n=6)
    dual_exps = explanation_engine.explain_dual_sourcing(opt_res, top_n=6)
    constraint_exps = explanation_engine.explain_binding_constraints(opt_res)
    all_exps = xfer_exps + dual_exps + constraint_exps

    record_explanations(conn, run_id_opt, all_exps)
    executive_narrative = explanation_engine.generate_executive_narrative(opt_res, base_res)

    conn.close()

    # 7. Serialize Artifacts to disk for UI and downstream consumption
    opt_res.orders.to_parquet(output_dir / "optimized_orders.parquet", index=False)
    opt_res.transfers.to_parquet(output_dir / "optimized_transfers.parquet", index=False)
    opt_res.shortages.to_parquet(output_dir / "optimized_shortages.parquet", index=False)
    opt_res.inventory.to_parquet(output_dir / "optimized_inventory.parquet", index=False)
    scenario_summary_df.to_parquet(output_dir / "scenario_comparison.parquet", index=False)
    comparison["cost_waterfall"].to_parquet(output_dir / "cost_waterfall.parquet", index=False)

    summary_payload = {
        "run_id": run_id_opt,
        "decision_date": str(latest_date),
        "status": opt_res.status,
        "solve_time_seconds": opt_res.solve_time_seconds,
        "baseline_total_cost": base_res.total_cost,
        "optimized_total_cost": opt_res.total_cost,
        "cost_difference": comparison["cost_difference"],
        "cost_reduction_pct": comparison["cost_reduction_pct"],
        "service_level": opt_res.service_level,
        "total_demand": opt_res.total_demand,
        "total_fulfilled": opt_res.total_fulfilled,
        "total_ordered": opt_res.total_ordered,
        "total_transferred": opt_res.total_transferred,
        "transfers_count": len(opt_res.transfers),
        "executive_narrative": executive_narrative,
        "cost_breakdown_optimized": opt_res.cost_breakdown,
        "cost_breakdown_baseline": base_res.cost_breakdown,
    }

    with open(output_dir / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("Prescriptive optimization execution and serialization completed successfully.")

    # Executive Output to Console
    print("\n" + "=" * 80)
    print("PRESCRIPTIVE OPERATIONS OPTIMIZATION EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Decision Date              : {latest_date}")
    print(f"Solver Engine              : SciPy HiGHS Dual Simplex (Continuous LP)")
    print(f"Solve Time                 : {opt_res.solve_time_seconds:.4f} seconds")
    print(f"Solver Status              : {opt_res.status}")
    print("-" * 80)
    print(f"Baseline Heuristic Cost    : ${base_res.total_cost:>12,.2f}  |  Fill Rate: {base_res.service_level*100:.1f}%")
    print(f"Prescriptive Optimal Cost  : ${opt_res.total_cost:>12,.2f}  |  Fill Rate: {opt_res.service_level*100:.1f}%")
    print(f"Modeled Cost Difference    : -${comparison['cost_difference']:>11,.2f} ({comparison['cost_reduction_pct']:.1f}% lower)")
    print(f"Lateral Transshipments     : {opt_res.total_transferred:>12,.0f} units across {len(opt_res.transfers)} lanes")
    print(f"Replenishment Orders       : {opt_res.total_ordered:>12,.0f} units")
    print("=" * 80)
    print(f"Executive Narrative:\n{executive_narrative}")
    print("=" * 80 + "\n")

    return summary_payload
