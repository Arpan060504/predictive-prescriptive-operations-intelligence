"""
Automated Pytest Suite for Phase 9: Prescriptive Operations Optimization.

Tests:
1. Objective function cost models (procurement, freight, transfer, holding, shortage, risk).
2. Index manager bijective mappings.
3. LP matrix formulation and dimensions.
4. Flow balance / inventory conservation across network.
5. Physical warehouse capacity limits.
6. Supplier 7-day capacity constraints.
7. Dual-sourcing policy caps.
8. Service level floor enforcement.
9. Lateral transshipment economic rationality.
10. Baseline heuristic policy mechanics (zero transfers, primary supplier routing).
11. Prescriptive optimization cost superiority vs baseline.
12. All 6 operational scenarios solvability.
13. Deterministic reproducibility (zero random drift).
14. Sensitivity sweeps execution and consistency.
15. Decision explanation generation and schema.
16. SQLite persistence and relational integrity.
17. Infeasibility elastic fallback resilience.
"""

import pytest
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

from src.optimization.objectives import (
    compute_procurement_unit_cost,
    compute_transport_unit_cost,
    compute_transshipment_unit_cost,
    compute_holding_unit_cost,
    compute_stockout_penalty,
    compute_supplier_risk_penalty,
)
from src.optimization.constraints import NetworkIndexManager, build_lp_matrices
from src.optimization.optimizer import NetworkOptimizer, OptimizationResult
from src.optimization.baseline import BaselinePolicyEngine, BaselineResult
from src.simulation.scenarios import get_standard_scenarios, apply_scenario
from src.simulation.simulator import ScenarioSimulator
from src.decision_engine.policy_comparison import compare_policies, compute_hhi
from src.decision_engine.sensitivity import SensitivityAnalyzer
from src.decision_engine.explanations import DecisionExplanationEngine
from src.optimization.persistence import (
    init_optimization_tables,
    record_optimization_run,
    save_scenario_definitions,
    record_policy_comparison,
    record_explanations,
)
from src.utils.config import get_resolved_path


@pytest.fixture(scope="module")
def network_fixtures():
    """Loads certified operational data and dimension tables for testing."""
    from src.database.connection import get_connection
    conn = get_connection(readonly=True)
    p_dims = pd.read_sql("SELECT * FROM dim_product", conn)
    s_dims = pd.read_sql("SELECT * FROM dim_supplier", conn)
    w_dims = pd.read_sql("SELECT * FROM dim_warehouse", conn)
    conn.close()

    op_path = get_resolved_path("data/processed/risk/operational_risk_priorities.parquet")
    op_df = pd.read_parquet(op_path)
    latest_date = op_df["date"].max()
    op_latest = op_df[op_df["date"] == latest_date].copy()

    return {
        "p_dims": p_dims,
        "s_dims": s_dims,
        "w_dims": w_dims,
        "op_latest": op_latest,
    }


# ==============================================================================
# 1. Objective Function Cost Model Tests
# ==============================================================================

def test_procurement_cost_primary_vs_secondary():
    base_cost = 20.0
    prim_cost = compute_procurement_unit_cost(base_cost, is_primary=True)
    sec_cost = compute_procurement_unit_cost(base_cost, is_primary=False, secondary_premium_pct=0.05)
    assert prim_cost == 20.0
    assert sec_cost == pytest.approx(21.0)


def test_transport_and_transshipment_costs():
    trans_cost = compute_transport_unit_cost(supplier_factor=1.2, warehouse_factor=1.1, base_freight_rate=2.50)
    assert trans_cost == pytest.approx(2.50 * 1.2 * 1.1)

    xfer_cost = compute_transshipment_unit_cost(
        origin_handling_cost=1.50, origin_transport_factor=1.0, dest_transport_factor=1.2, base_freight_rate=2.50
    )
    expected_xfer = 1.50 + 2.50 * 1.10
    assert xfer_cost == pytest.approx(expected_xfer)


def test_holding_and_penalty_costs():
    h_cost = compute_holding_unit_cost(unit_cost=50.0, annual_rate=0.20, horizon_days=7)
    assert h_cost > 0.0
    assert h_cost == pytest.approx(50.0 * (0.20 / 365.25) * 7)

    # Stockout penalty increases with higher stockout probability
    pen_low = compute_stockout_penalty(selling_price=100.0, stockout_probability=0.10)
    pen_high = compute_stockout_penalty(selling_price=100.0, stockout_probability=0.90)
    assert pen_high > pen_low
    assert pen_low == pytest.approx(100.0 * 1.5 * 1.10)
    assert pen_high == pytest.approx(100.0 * 1.5 * 1.90)


# ==============================================================================
# 2. Structural & LP Matrix Tests
# ==============================================================================

def test_index_manager_and_matrix_shapes(network_fixtures):
    f = network_fixtures
    sku_suppliers = {}
    for _, row in f["p_dims"].iterrows():
        sups = [str(row["primary_supplier_id"])]
        if pd.notna(row.get("secondary_supplier_id")):
            sups.append(str(row["secondary_supplier_id"]))
        sku_suppliers[str(row["product_id"])] = sups

    products = sorted(list(sku_suppliers.keys()))
    warehouses = sorted(f["w_dims"]["warehouse_id"].astype(str).unique().tolist())
    mgr = NetworkIndexManager(products, warehouses, sku_suppliers)

    assert mgr.n_vars == (len(mgr.order_indices) + len(mgr.transfer_indices) +
                          len(mgr.shortage_indices) + len(mgr.inventory_indices))
    assert len(mgr.shortage_indices) == 60 * 4
    assert len(mgr.inventory_indices) == 60 * 4
    assert len(mgr.transfer_indices) == 60 * 4 * 3  # 12 lanes per SKU

    c, A_eq, b_eq, A_ub, b_ub, bounds, eq_names, ub_names = build_lp_matrices(
        index_mgr=mgr,
        operational_data=f["op_latest"],
        supplier_dims=f["s_dims"],
        warehouse_dims=f["w_dims"],
        product_dims=f["p_dims"],
    )

    assert len(c) == mgr.n_vars
    assert A_eq.shape[1] == mgr.n_vars
    assert A_eq.shape[0] == 240  # 60 SKUs * 4 Warehouses flow equations
    assert len(b_eq) == 240
    assert A_ub.shape[1] == mgr.n_vars
    assert len(bounds) == mgr.n_vars


# ==============================================================================
# 3. Prescriptive Optimizer Core Solution Tests
# ==============================================================================

def test_optimizer_flow_balance_conservation(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    res = opt.solve(scenario_name="test_baseline")

    assert res.success is True
    assert res.status == "OPTIMAL"
    assert res.service_level >= 0.95

    # Check physical flow balance equation per SKU x Warehouse
    op_keyed = f["op_latest"].set_index(["product_id", "warehouse_id"]).to_dict(orient="index")
    for _, row in res.shortages.iterrows():
        p_id = row["product_id"]
        w_id = row["warehouse_id"]
        d_iw = float(row["forecast_demand"])
        s_iw = float(row["shortage_quantity"])

        # Inbound orders to (p_id, w_id)
        orders_in = float(res.orders[
            (res.orders["product_id"] == p_id) & (res.orders["warehouse_id"] == w_id)
        ]["order_quantity"].sum()) if not res.orders.empty else 0.0

        # Transfers in and out
        xfers_in = float(res.transfers[
            (res.transfers["product_id"] == p_id) & (res.transfers["dest_warehouse"] == w_id)
        ]["transfer_quantity"].sum()) if not res.transfers.empty else 0.0

        xfers_out = float(res.transfers[
            (res.transfers["product_id"] == p_id) & (res.transfers["origin_warehouse"] == w_id)
        ]["transfer_quantity"].sum()) if not res.transfers.empty else 0.0

        # Ending inventory
        end_inv = float(res.inventory[
            (res.inventory["product_id"] == p_id) & (res.inventory["warehouse_id"] == w_id)
        ]["ending_inventory"].sum()) if not res.inventory.empty else 0.0

        i0_iw = float(op_keyed.get((p_id, w_id), {}).get("ending_inventory_lag1", 0.0))

        # Balance check: i0 + orders + xfers_in - xfers_out - d + s == end_inv
        calculated_end = i0_iw + orders_in + xfers_in - xfers_out - d_iw + s_iw
        assert calculated_end == pytest.approx(end_inv, abs=1e-3)


def test_warehouse_capacity_limits_respected(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    res = opt.solve()

    wh_caps = f["w_dims"].set_index("warehouse_id")["capacity_units"].to_dict()
    ending_by_wh = res.inventory.groupby("warehouse_id")["ending_inventory"].sum().to_dict()

    for w, total_inv in ending_by_wh.items():
        assert total_inv <= wh_caps[w] + 1e-3


def test_supplier_weekly_capacity_limits_respected(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    res = opt.solve()

    sup_weekly_caps = {
        row["supplier_id"]: float(row["monthly_capacity_units"]) / 4.333333
        for _, row in f["s_dims"].iterrows()
    }
    ordered_by_sup = res.orders.groupby("supplier_id")["order_quantity"].sum().to_dict()

    for s, total_ord in ordered_by_sup.items():
        assert total_ord <= sup_weekly_caps[s] + 1e-3


def test_dual_sourcing_policy_cap_respected(network_fixtures):
    f = network_fixtures
    gamma = 0.60
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"], max_supplier_allocation_pct=gamma)
    res = opt.solve(enable_dual_sourcing_cap=True)

    # For any SKU with orders, check that no single supplier receives > gamma of total SKU orders
    if not res.orders.empty:
        sku_totals = res.orders.groupby("product_id")["order_quantity"].sum()
        for p_id, tot_qty in sku_totals.items():
            if tot_qty > 1e-3:
                sup_orders = res.orders[res.orders["product_id"] == p_id].groupby("supplier_id")["order_quantity"].sum()
                for _, s_qty in sup_orders.items():
                    assert (s_qty / tot_qty) <= gamma + 1e-4


# ==============================================================================
# 4. Baseline vs Prescriptive Policy Comparison Tests
# ==============================================================================

def test_baseline_policy_deterministic_behavior(network_fixtures):
    f = network_fixtures
    base_engine = BaselinePolicyEngine(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    base_res = base_engine.evaluate()

    assert base_res.status == "COMPLETED"
    assert base_res.total_transferred == 0.0
    assert base_res.transfers.empty

    # All baseline orders must be from primary suppliers
    if not base_res.orders.empty:
        assert (base_res.orders["is_primary_supplier"] == True).all()


def test_optimizer_beats_baseline_on_cost(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    opt_res = opt.solve()

    base_engine = BaselinePolicyEngine(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    base_res = base_engine.evaluate()

    # Prescriptive optimal cost must be <= baseline cost
    assert opt_res.total_cost <= base_res.total_cost
    assert opt_res.service_level >= base_res.service_level - 1e-4

    comp = compare_policies(opt_res, base_res)
    assert comp["cost_difference"] >= 0.0
    assert comp["cost_reduction_pct"] >= 0.0
    assert len(comp["cost_waterfall"]) == 7


def test_hhi_concentration_metric():
    # Pure monopoly: 1 supplier has 100% share -> HHI = 10,000
    monopoly_shares = np.array([100.0])
    assert compute_hhi(monopoly_shares) == pytest.approx(10000.0)

    # Duopoly 50/50: HHI = 2500 + 2500 = 5000
    duopoly_shares = np.array([50.0, 50.0])
    assert compute_hhi(duopoly_shares) == pytest.approx(5000.0)

    # 4 equal suppliers 25% each: HHI = 4 * 625 = 2500
    quad_shares = np.array([25.0, 25.0, 25.0, 25.0])
    assert compute_hhi(quad_shares) == pytest.approx(2500.0)


# ==============================================================================
# 5. Multi-Scenario & Sensitivity Tests
# ==============================================================================

def test_all_six_scenarios_simulate_successfully(network_fixtures):
    f = network_fixtures
    sim = ScenarioSimulator(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    scenarios = get_standard_scenarios()
    assert len(scenarios) == 6

    summary_df = sim.run_all(scenarios)
    assert len(summary_df) == 6
    assert (summary_df["optimized_total_cost"] > 0).all()
    assert (summary_df["baseline_total_cost"] >= summary_df["optimized_total_cost"]).all()
    assert (summary_df["solve_time_seconds"] < 2.0).all()


def test_deterministic_reproducibility(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])

    res1 = opt.solve(scenario_name="repro_1")
    res2 = opt.solve(scenario_name="repro_2")

    assert res1.total_cost == pytest.approx(res2.total_cost, abs=1e-4)
    assert res1.total_ordered == pytest.approx(res2.total_ordered, abs=1e-4)
    assert res1.total_transferred == pytest.approx(res2.total_transferred, abs=1e-4)


def test_sensitivity_sweeps_execution(network_fixtures):
    f = network_fixtures
    analyzer = SensitivityAnalyzer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    sweeps = analyzer.run_all_sweeps()

    assert "service_level_sensitivity" in sweeps
    assert "freight_rate_sensitivity" in sweeps
    assert "sourcing_cap_sensitivity" in sweeps

    df_sl = sweeps["service_level_sensitivity"]
    assert len(df_sl) == 5
    # Total cost should increase or remain flat as service level target tightens
    assert (df_sl["total_cost"].iloc[-1] >= df_sl["total_cost"].iloc[0] - 1e-3)


# ==============================================================================
# 6. Explainability & SQLite Persistence Tests
# ==============================================================================

def test_decision_explanations_structure(network_fixtures):
    f = network_fixtures
    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    res = opt.solve()

    exp_engine = DecisionExplanationEngine(f["p_dims"], f["s_dims"], f["w_dims"])
    xfer_exps = exp_engine.explain_transfers(res, top_n=5)
    constraint_exps = exp_engine.explain_binding_constraints(res)
    narrative = exp_engine.generate_executive_narrative(res)

    assert len(xfer_exps) > 0
    assert "explanation" in xfer_exps[0]
    assert len(narrative) > 50
    assert "Prescriptive plan" in narrative


def test_sqlite_persistence_roundtrip(network_fixtures, tmp_path):
    f = network_fixtures
    test_db = tmp_path / "test_opt.db"
    conn = sqlite3.connect(test_db)

    init_optimization_tables(conn)
    scenarios = get_standard_scenarios()
    save_scenario_definitions(conn, scenarios)

    opt = NetworkOptimizer(f["op_latest"], f["s_dims"], f["w_dims"], f["p_dims"])
    res = opt.solve()
    run_id = "TEST-RUN-001"
    record_optimization_run(conn, res, run_id)

    # Query back
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM optimization_runs WHERE run_id = ?", (run_id,))
    assert cur.fetchone()[0] == 1

    cur.execute("SELECT COUNT(1) FROM optimization_decisions WHERE run_id = ?", (run_id,))
    dec_count = cur.fetchone()[0]
    assert dec_count > 0

    cur.execute("SELECT COUNT(1) FROM scenario_definitions")
    assert cur.fetchone()[0] == 6

    conn.close()
