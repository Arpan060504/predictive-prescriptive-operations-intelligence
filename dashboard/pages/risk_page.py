"""
Inventory & Supplier Risk Intelligence Streamlit Page for PPOI (Phase 8).
Renders executive KPIs, probability distributions, exposure matrix,
investigation queue, and drill-down risk explanations.
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from src.database.connection import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_data(ttl=600)
def load_risk_data():
    """Loads certified risk predictions and exposure table from parquet/SQLite."""
    risk_dir = PROJECT_ROOT / "data" / "processed" / "risk"
    exp_path = risk_dir / "operational_risk_priorities.parquet"
    sup_path = risk_dir / "supplier_risk_predictions.parquet"
    
    if exp_path.exists():
        df_exp = pd.read_parquet(exp_path)
    else:
        conn = get_connection(readonly=True)
        df_exp = pd.read_sql_query("SELECT * FROM analytics_operational_exposure", conn)
        conn.close()
        
    if sup_path.exists():
        df_sup = pd.read_parquet(sup_path)
    else:
        conn = get_connection(readonly=True)
        df_sup = pd.read_sql_query("SELECT * FROM analytics_supplier_risk", conn)
        conn.close()
        
    return df_exp, df_sup


def render_risk_page():
    st.title("🛡️ Inventory & Supplier Risk Intelligence")
    st.markdown("Transforming multi-horizon predictive forecasts and supplier reliability into explainable operational prioritization.")
    
    try:
        df_exp, df_sup = load_risk_data()
    except Exception as e:
        st.error(f"Error loading risk data: {e}. Please ensure `python run.py train-risk-models` has run.")
        return

    # Filter to latest date snapshot for executive cross-section
    latest_date = df_exp["date"].max()
    latest_snapshot = df_exp[df_exp["date"] == latest_date].copy()

    # -------------------------------------------------------------------------
    # 1. Executive Risk Summary KPIs
    # -------------------------------------------------------------------------
    st.subheader(f"1. Executive Risk Snapshot (As of {latest_date})")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    high_inv_count = int((latest_snapshot["inventory_risk_score"] >= 60.0).sum())
    total_positions = len(latest_snapshot)
    avg_stockout_prob = float(latest_snapshot["stockout_probability_7d"].mean() * 100.0)
    total_proj_exposure = float(latest_snapshot["projected_exposure_cost"].sum())
    
    latest_sup_snapshot = df_sup.sort_values("order_date").groupby("supplier_id").last().reset_index()
    high_sup_count = int((latest_sup_snapshot["supplier_risk_score"] >= 60.0).sum())
    total_suppliers = len(latest_sup_snapshot)
    
    critical_inv_value = float(
        (latest_snapshot[latest_snapshot["operational_exposure_band"].isin(["HIGH", "CRITICAL"])]["projected_exposure_cost"]).sum()
    )

    with kpi_col1:
        st.metric(
            label="High-Risk Positions",
            value=f"{high_inv_count} / {total_positions}",
            delta=f"{(high_inv_count/total_positions)*100:.1f}% of network",
            delta_color="inverse"
        )
    with kpi_col2:
        st.metric(
            label="Elevated-Risk Suppliers",
            value=f"{high_sup_count} / {total_suppliers}",
            delta="Tier 1/2 Sourcing",
            delta_color="off"
        )
    with kpi_col3:
        st.metric(
            label="Avg 7d Stockout Prob",
            value=f"{avg_stockout_prob:.1f}%",
            delta="Calibrated XGBoost",
            delta_color="off"
        )
    with kpi_col4:
        st.metric(
            label="Total Projected Exposure",
            value=f"${total_proj_exposure:,.0f}",
            delta="1.5x penalty adjusted",
            delta_color="inverse"
        )
    with kpi_col5:
        st.metric(
            label="Critical At-Risk Value",
            value=f"${critical_inv_value:,.0f}",
            delta="Immediate Attention",
            delta_color="inverse"
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Inventory Risk Analytics
    # -------------------------------------------------------------------------
    st.subheader("2. Inventory Stockout Risk Distribution")
    tab_inv1, tab_inv2 = st.columns(2)
    
    with tab_inv1:
        st.markdown("**Stockout Probability Distribution (7-Day Horizon)**")
        prob_hist, bin_edges = np.histogram(latest_snapshot["stockout_probability_7d"], bins=10, range=(0, 1))
        hist_df = pd.DataFrame({
            "Probability Range": [f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}" for i in range(len(prob_hist))],
            "Count": prob_hist
        })
        st.bar_chart(hist_df.set_index("Probability Range"))
        
    with tab_inv2:
        st.markdown("**Exposure Bands by Warehouse**")
        wh_band = pd.crosstab(latest_snapshot["warehouse_id"], latest_snapshot["operational_exposure_band"])
        st.bar_chart(wh_band)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Supplier Delay Risk Analytics
    # -------------------------------------------------------------------------
    st.subheader("3. Supplier Delay Risk & Reliability Trend")
    sup_col1, sup_col2 = st.columns(2)
    
    with sup_col1:
        st.markdown("**Supplier Risk Profile & Historical Reliability**")
        sup_display = latest_sup_snapshot[[
            "supplier_id", "supplier_name", "supplier_tier",
            "supplier_delay_probability", "supplier_risk_score", "supplier_risk_band",
            "hist_on_time_rate", "hist_p90_delay"
        ]].copy()
        sup_display["supplier_delay_probability"] = (sup_display["supplier_delay_probability"] * 100).round(1).astype(str) + "%"
        sup_display["hist_on_time_rate"] = (sup_display["hist_on_time_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(sup_display.sort_values("supplier_risk_score", ascending=False), use_container_width=True, hide_index=True)
        
    with sup_col2:
        st.markdown("**Supplier Capacity Pressure vs P90 Delay (Days)**")
        chart_data = latest_sup_snapshot.set_index("supplier_id")[["supplier_capacity_pressure", "hist_p90_delay"]]
        st.line_chart(chart_data)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. SKU × Warehouse Operational Exposure Matrix
    # -------------------------------------------------------------------------
    st.subheader("4. SKU × Warehouse Operational Exposure Matrix")
    st.caption("Normalized Operational Exposure Score (0-100) combining demand forecast, buffer deficit, and supplier risk.")
    
    matrix = latest_snapshot.pivot(index="product_id", columns="warehouse_id", values="operational_exposure_score")
    st.dataframe(matrix.style.background_gradient(cmap="YlOrRd", vmin=10, vmax=90), use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5. Prioritized Investigation Queue
    # -------------------------------------------------------------------------
    st.subheader("5. Prioritized Operational Investigation Queue")
    
    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_band = st.multiselect("Filter by Exposure Band", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
    with f_col2:
        sel_wh = st.multiselect("Filter by Warehouse", latest_snapshot["warehouse_id"].unique(), default=latest_snapshot["warehouse_id"].unique())
    with f_col3:
        sel_cat = st.multiselect("Filter by Category", latest_snapshot["category"].unique(), default=latest_snapshot["category"].unique())

    filtered_queue = latest_snapshot[
        (latest_snapshot["operational_exposure_band"].isin(sel_band)) &
        (latest_snapshot["warehouse_id"].isin(sel_wh)) &
        (latest_snapshot["category"].isin(sel_cat))
    ].sort_values("operational_exposure_score", ascending=False)

    queue_display_cols = [
        "product_id", "category", "warehouse_id", "primary_supplier_name",
        "forecast_demand_7d", "inventory_position", "inventory_days_of_supply",
        "stockout_probability_7d", "supplier_delay_probability",
        "operational_exposure_score", "operational_exposure_band",
        "explanation_summary", "recommended_action"
    ]
    st.dataframe(filtered_queue[queue_display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Entity Drill-Down & Explainability Inspector
    # -------------------------------------------------------------------------
    st.subheader("6. Entity Drill-Down & Deterministic Explanation Inspector")
    
    selected_sku = st.selectbox("Select SKU for Deep-Dive Audit:", sorted(filtered_queue["product_id"].unique()) if len(filtered_queue) > 0 else sorted(latest_snapshot["product_id"].unique()))
    selected_wh = st.selectbox("Select Warehouse:", sorted(latest_snapshot[latest_snapshot["product_id"] == selected_sku]["warehouse_id"].unique()))

    record = latest_snapshot[(latest_snapshot["product_id"] == selected_sku) & (latest_snapshot["warehouse_id"] == selected_wh)].iloc[0]

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown(f"### Position: `{selected_sku}` @ `{selected_wh}`")
        st.markdown(f"**Product Name:** {record.get('product_name', 'N/A')}")
        st.markdown(f"**Category:** {record.get('category', 'N/A')} | **Criticality:** `{record.get('criticality', 'N/A')}`")
        st.markdown(f"**Primary Supplier:** {record.get('primary_supplier_name', 'N/A')} (`{record.get('primary_supplier_id', 'N/A')}`)")
        st.markdown(f"**Supplier Dependency:** {record.get('supplier_dependency', 'Standard')}")
        
        st.markdown("#### Operational Metrics:")
        st.write({
            "Forecast 7d Demand": f"{record['forecast_demand_7d']:.1f} units",
            "Available Inventory": f"{record['inventory_position']:.1f} units",
            "Days of Supply": f"{record['inventory_days_of_supply']:.1f} days",
            "Stockout Probability (7d)": f"{record['stockout_probability_7d']*100:.1f}%",
            "Supplier Delay Probability": f"{record['supplier_delay_probability']*100:.1f}%",
            "Inventory Risk Score": f"{record['inventory_risk_score']:.1f} / 100",
            "Supplier Risk Score": f"{record['supplier_risk_score']:.1f} / 100",
            "Operational Exposure Score": f"{record['operational_exposure_score']:.1f} / 100",
            "Exposure Band": record['operational_exposure_band'],
            "Projected Exposure Value": f"${record['projected_exposure_cost']:,.2f}",
        })
        
    with d_col2:
        st.markdown("### 🔍 Deterministic Audit Evidence")
        st.info(f"**Summary:** {record['explanation_summary']}")
        
        st.markdown("**Evidence Bullet Points (Traceable to Features):**")
        ev_items = str(record['evidence_bullets']).split(" | ")
        for ev in ev_items:
            st.markdown(f"- {ev}")
            
        st.markdown("#### 🎯 Prescriptive Recommended Action:")
        st.warning(f"**Action:** {record['recommended_action']}")
        st.caption("Model Version: XGBoost-v1.0 (Inventory) & RandomForest-v1.0 (Supplier) calibrated with Sigmoid Scaling.")
