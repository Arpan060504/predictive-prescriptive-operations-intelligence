"""
Prescriptive Operations Optimization Streamlit Page for PPOI (Phase 9).

Interactive Prescriptive Operations Center:
- Scenario Selector (Baseline + 5 Stress Scenarios)
- Executive KPI Cards & Modeled Landed Cost Differences
- Cost Breakdown Waterfall (Procurement, Freight, Transshipments, Holding, Shortage, Risk)
- Prescriptive Order Allocations & Supplier Sourcing Splits (Dual-Sourcing HHI)
- Multi-Echelon Lateral Transshipment Rebalancing Flow Table
- Binding Constraints & Dual Variable (Shadow Price) Inspector
- Deterministic Decision Explainability Drawer & Executive Narrative
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.optimization.optimizer import NetworkOptimizer, OptimizationResult
from src.optimization.baseline import BaselinePolicyEngine, BaselineResult
from src.simulation.scenarios import get_standard_scenarios, apply_scenario, ScenarioDefinition
from src.decision_engine.policy_comparison import compare_policies
from src.decision_engine.explanations import DecisionExplanationEngine
from src.utils.config import get_resolved_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_data(ttl=600)
def load_prescriptive_base_data():
    """Loads certified operational data and dimension tables."""
    db_path = get_resolved_path("database/operations.db")
    conn = sqlite3.connect(db_path)
    p_dims = pd.read_sql("SELECT * FROM dim_product", conn)
    s_dims = pd.read_sql("SELECT * FROM dim_supplier", conn)
    w_dims = pd.read_sql("SELECT * FROM dim_warehouse", conn)
    conn.close()

    op_path = get_resolved_path("data/processed/risk/operational_risk_priorities.parquet")
    op_df = pd.read_parquet(op_path)
    latest_date = op_df["date"].max()
    op_latest = op_df[op_df["date"] == latest_date].copy()

    return op_latest, p_dims, s_dims, w_dims, latest_date


@st.cache_data(ttl=600)
def run_scenario_prescriptive(scenario_name: str):
    """Executes optimizer and baseline for selected scenario with caching."""
    op_latest, p_dims, s_dims, w_dims, _ = load_prescriptive_base_data()
    scenarios = {s.name: s for s in get_standard_scenarios()}
    sc = scenarios.get(scenario_name, scenarios["baseline"])

    op_sc, sup_sc, wh_sc = apply_scenario(op_latest, s_dims, w_dims, sc)

    # 1. Baseline
    base_engine = BaselinePolicyEngine(op_sc, sup_sc, wh_sc, p_dims, freight_rate_base=2.50)
    base_res = base_engine.evaluate(scenario_name=sc.name, freight_multiplier=sc.freight_multiplier)

    # 2. Optimizer
    optimizer = NetworkOptimizer(
        op_sc, sup_sc, wh_sc, p_dims,
        service_level_target=sc.service_level_target,
        freight_rate_base=2.50,
    )
    opt_res = optimizer.solve(
        scenario_name=sc.name,
        freight_multiplier=sc.freight_multiplier,
        enable_transfers=True,
        enable_dual_sourcing_cap=True,
    )

    comparison = compare_policies(opt_res, base_res)
    exp_engine = DecisionExplanationEngine(p_dims, sup_sc, wh_sc)
    narrative = exp_engine.generate_executive_narrative(opt_res, base_res)
    xfer_exps = exp_engine.explain_transfers(opt_res, top_n=6)
    dual_exps = exp_engine.explain_dual_sourcing(opt_res, top_n=6)
    constraint_exps = exp_engine.explain_binding_constraints(opt_res)

    return opt_res, base_res, comparison, narrative, xfer_exps, dual_exps, constraint_exps


def render_prescriptive_page():
    st.title("⚖️ Prescriptive Operations Optimization")
    st.markdown(
        "Constrained multi-echelon network optimization engine solving replenishment, "
        "risk-weighted supplier dual-sourcing, and lateral inventory rebalancing."
    )

    try:
        op_latest, p_dims, s_dims, w_dims, latest_date = load_prescriptive_base_data()
    except Exception as e:
        st.error(f"Error loading optimization prerequisites: {e}")
        return

    # Scenario Selection
    scenarios = get_standard_scenarios()
    sc_map = {s.display_name: s.name for s in scenarios}
    selected_display = st.selectbox(
        "Select Operational Scenario Simulation:",
        options=list(sc_map.keys()),
        index=0,
    )
    selected_scenario_name = sc_map[selected_display]
    current_sc_def = next(s for s in scenarios if s.name == selected_scenario_name)

    st.caption(f"**Scenario Description:** {current_sc_def.description}")

    # Run / Retrieve Prescriptive Solution
    with st.spinner("Solving Prescriptive Network Optimization Model (HiGHS)..."):
        opt_res, base_res, comparison, narrative, xfer_exps, dual_exps, constraint_exps = run_scenario_prescriptive(
            selected_scenario_name
        )

    # -------------------------------------------------------------------------
    # 1. Executive Telemetry & KPI Cards
    # -------------------------------------------------------------------------
    st.subheader(f"1. Executive Operational Telemetry (Decision Date: {latest_date})")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            label="Prescriptive Landed Cost",
            value=f"${opt_res.total_cost:,.0f}",
            delta=f"-${comparison['cost_difference']:,.0f} Modeled Opex Diff",
            delta_color="normal",
        )
    with kpi2:
        st.metric(
            label="Baseline Heuristic Cost",
            value=f"${base_res.total_cost:,.0f}",
            delta="Status-Quo Heuristic",
            delta_color="off",
        )
    with kpi3:
        st.metric(
            label="Network Fill Rate",
            value=f"{opt_res.service_level*100:.1f}%",
            delta=f"Baseline: {base_res.service_level*100:.1f}%",
            delta_color="normal" if opt_res.service_level >= base_res.service_level else "inverse",
        )
    with kpi4:
        st.metric(
            label="Lateral Transshipments",
            value=f"{opt_res.total_transferred:,.0f} units",
            delta=f"{len(opt_res.transfers)} active transfer lanes",
            delta_color="off",
        )
    with kpi5:
        st.metric(
            label="HiGHS Solve Latency",
            value=f"{opt_res.solve_time_seconds*1000:.1f} ms",
            delta=f"Status: {opt_res.status}",
            delta_color="off",
        )

    # Executive Narrative Alert Box
    st.info(f"**Operational Intelligence Synthesis:** {narrative}")

    # Accounting Note on Inventory Disparity
    st.warning(
        "**Accounting note:** The baseline ends the 7-day horizon with substantially more inventory. "
        "Therefore modeled expenditure differences should not be interpreted as equivalent to long-term economic savings."
    )

    # -------------------------------------------------------------------------
    # 2. Cost Waterfall & Trade-off Visualization
    # -------------------------------------------------------------------------
    st.subheader("2. Operational Cost Breakdown & Trade-Off Analysis")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        # Cost Waterfall / Comparison Bar Chart
        categories = ["Procurement", "Inbound Freight", "Transshipment", "Holding", "Delay Risk", "Total Landed"]
        base_vals = [
            base_res.cost_breakdown.get("procurement_cost", 0.0),
            base_res.cost_breakdown.get("transport_cost", 0.0),
            base_res.cost_breakdown.get("transshipment_cost", 0.0),
            base_res.cost_breakdown.get("holding_cost", 0.0),
            base_res.cost_breakdown.get("supplier_risk_cost", 0.0),
            base_res.total_cost,
        ]
        opt_vals = [
            opt_res.cost_breakdown.get("procurement_cost", 0.0),
            opt_res.cost_breakdown.get("transport_cost", 0.0),
            opt_res.cost_breakdown.get("transshipment_cost", 0.0),
            opt_res.cost_breakdown.get("holding_cost", 0.0),
            opt_res.cost_breakdown.get("supplier_risk_cost", 0.0),
            opt_res.total_cost,
        ]

        fig_cost = go.Figure(data=[
            go.Bar(name="Baseline Heuristic", x=categories, y=base_vals, marker_color="#94a3b8"),
            go.Bar(name="Prescriptive Optimal", x=categories, y=opt_vals, marker_color="#2563eb"),
        ])
        fig_cost.update_layout(
            title="Cost Component Comparison ($)",
            barmode="group",
            height=360,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_chart2:
        # Sourcing Concentration (HHI & Supplier Shares)
        df_sourcing = comparison["sourcing_breakdown"]
        fig_src = go.Figure(data=[
            go.Bar(name="Baseline Order Share (%)", x=df_sourcing["supplier_id"], y=df_sourcing["baseline_share_pct"], marker_color="#cbd5e1"),
            go.Bar(name="Optimized Order Share (%)", x=df_sourcing["supplier_id"], y=df_sourcing["optimized_share_pct"], marker_color="#0d9488"),
        ])
        fig_src.update_layout(
            title=f"Supplier Sourcing Shares (Network HHI: {comparison['baseline_hhi']} → {comparison['optimized_hhi']})",
            barmode="group",
            height=360,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_src, use_container_width=True)
        st.caption("Higher HHI = greater concentration. Although the optimizer satisfies the 60% maximum single-supplier constraint at the SKU level, the resulting network-level sourcing portfolio is more concentrated than the baseline.")

    # -------------------------------------------------------------------------
    # 3. Prescriptive Decisions: Transfers & Orders
    # -------------------------------------------------------------------------
    st.subheader("3. Actionable Prescriptive Decision Tables")

    tab_xfers, tab_orders, tab_constraints = st.tabs([
        "🔄 Lateral Transshipments (Network Rebalancing)",
        "📦 Replenishment Purchase Orders",
        "⛓️ Binding Constraints & Shadow Prices",
    ])

    with tab_xfers:
        st.markdown(
            "Lateral transfers mobilize idle stock from surplus distribution centers to deficit regions, "
            "avoiding expensive expedited purchase orders and supplier lead time exposure."
        )
        if not opt_res.transfers.empty:
            st.dataframe(
                opt_res.transfers.style.format({
                    "transfer_quantity": "{:,.0f}",
                    "unit_transfer_cost": "${:.2f}",
                    "total_transfer_cost": "${:,.2f}",
                }),
                use_container_width=True,
                height=300,
            )
        else:
            st.info("No lateral transfers required under this operational scenario.")

    with tab_orders:
        st.markdown(
            "Optimal purchase orders allocated across qualified primary and secondary suppliers, "
            "balancing unit cost, freight logistics, delivery delay risk, and 60% diversification caps."
        )
        if not opt_res.orders.empty:
            st.dataframe(
                opt_res.orders.style.format({
                    "order_quantity": "{:,.0f}",
                    "procurement_cost": "${:,.2f}",
                    "transport_cost": "${:,.2f}",
                    "risk_cost": "${:,.2f}",
                    "total_order_cost": "${:,.2f}",
                }),
                use_container_width=True,
                height=300,
            )
        else:
            st.info("No new procurement orders required; network demand satisfied entirely via existing on-hand stock and lateral transfers.")

    with tab_constraints:
        st.markdown(
            "Active and binding constraints identified by HiGHS. Shadow prices indicate the exact marginal reduction "
            "in total landed cost achieved by relaxing a given operational bottleneck by 1 unit."
        )
        if not opt_res.constraints_summary.empty:
            binding_df = opt_res.constraints_summary[opt_res.constraints_summary["is_binding"] == True]
            if not binding_df.empty:
                st.dataframe(
                    binding_df.style.format({
                        "slack": "{:.2f}",
                        "shadow_price": "${:.4f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.success("No operational capacity constraints are binding at current demand levels.")
        else:
            st.info("Constraint summary not available.")

    # -------------------------------------------------------------------------
    # 4. Deterministic Explainability Drawer
    # -------------------------------------------------------------------------
    st.subheader("4. Deterministic Decision Explainability (Audit & Governance)")

    with st.expander("🔍 View Verifiable Decision Evidence & Rationales", expanded=True):
        if xfer_exps:
            st.markdown("##### Lateral Transshipment Rationales:")
            for exp in xfer_exps:
                st.markdown(f"- **{exp['entity_id']}**: {exp['explanation']}")

        if dual_exps:
            st.markdown("##### Sourcing Diversification & Cap Rationales:")
            for exp in dual_exps:
                st.markdown(f"- **{exp['product_id']}**: {exp['explanation']}")

        if constraint_exps:
            st.markdown("##### Constraint Bottlenecks & Shadow Value Rationales:")
            for exp in constraint_exps:
                st.markdown(f"- **{exp['entity_id']}**: {exp['explanation']}")

    # -------------------------------------------------------------------------
    # 5. Governance and Disclosures
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "**Governance & Methodology Disclosure:** The prescriptive optimization results are scenario-based and "
        "evaluated over a finite 7-day horizon. The optimized policy may consume existing network inventory rather than "
        "purchase additional safety stock. Consequently, modeled operational-expenditure differences should not be "
        "interpreted as realized financial savings. The current optimization reduces modeled supplier-delay exposure "
        "but can increase network-level supplier concentration despite satisfying the configured per-SKU supplier allocation cap."
    )
