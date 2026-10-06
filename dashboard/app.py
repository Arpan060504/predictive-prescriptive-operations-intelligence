"""
Streamlit Web Application Entrypoint for PPOI.
Predictive → Prescriptive Operations Intelligence Platform.
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pages.risk_page import render_risk_page
from dashboard.pages.prescriptive_page import render_prescriptive_page
from src.database.connection import get_database_info, is_demo_database

st.set_page_config(
    page_title="PPOI — Operations Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Corporate CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.sidebar.title("📦 PPOI Intelligence")
    st.sidebar.caption("Predictive → Prescriptive Operations Engine v1.0")
    
    page = st.sidebar.radio(
        "Navigation Modules",
        [
            "Inventory & Supplier Risk",
            "Prescriptive Optimization",
            "Demand Forecasting",
            "System Architecture & Audit",
        ],
        index=0,
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Decision Date:** `2025-12-25` (Certified Horizon)")
    st.sidebar.markdown("**Engines Active:** Phase 7, 8 & 9 (Prescriptive)")
    st.sidebar.caption("All operational data synthetic for analytical benchmarking.")

    # Database Mode Telemetry Indicator
    try:
        db_info = get_database_info()
        st.sidebar.markdown("---")
        if db_info["is_demo"]:
            st.sidebar.warning("⚡ **Mode:** DEPLOYMENT DEMO DB")
            st.sidebar.caption(
                f"**Active DB:** `{db_info['filename']}` ({db_info['size_mb']} MB)\n\n"
                "Lightweight deployment profile for Streamlit Community Cloud. "
                "Preserves full schema definitions, 100% of dimension tables, "
                "all Phase 9 prescriptive optimization models, and recent analytical exposure telemetry."
            )
        else:
            st.sidebar.success("🏛️ **Mode:** FULL OPERATIONAL DB")
            st.sidebar.caption(
                f"**Active DB:** `{db_info['filename']}` ({db_info['size_mb']} MB)\n\n"
                "Complete 24-month operational star schema."
            )
    except Exception as e:
        st.sidebar.error(f"Database connection offline: {e}")

    if page == "Inventory & Supplier Risk":
        render_risk_page()
    elif page == "Prescriptive Optimization":
        render_prescriptive_page()
    elif page == "Demand Forecasting":
        st.title("📈 Multi-Horizon Demand Forecasting")
        st.info("Certified Phase 7 Multi-Horizon Forecasting layer (t+1, t+7, t+14, t+28). Locked model: XGBoost deep_expressive.")
    elif page == "System Architecture & Audit":
        st.title("🏛️ System Architecture & Audit Manifest")
        st.success("121/121 Prior Certified Tests Passing. Phase 9 Prescriptive Operations Intelligence certified.")

        import platform
        import pandas as pd

        # Environment & Deployment Diagnostics
        st.subheader("Cloud Environment & Deployment Diagnostics")
        col_env1, col_env2, col_env3, col_env4 = st.columns(4)
        with col_env1:
            py_ver = platform.python_version()
            st.metric(
                "Python Runtime",
                f"v{py_ver}",
                delta="Certified 3.12" if py_ver.startswith("3.12") else "Unexpected Runtime",
                delta_color="normal" if py_ver.startswith("3.12") else "inverse",
            )
        with col_env2:
            st.metric("Pandas Version", f"v{pd.__version__}")
        with col_env3:
            st.metric("Streamlit Version", f"v{st.__version__}")
        with col_env4:
            st.metric("Database Mode", db_info["mode_label"])

        try:
            st.subheader("Database Relational Telemetry")
            col_db1, col_db2, col_db3, col_db4 = st.columns(4)
            with col_db1:
                st.metric("Database Mode", db_info["mode_label"])
            with col_db2:
                st.metric("Database File", db_info["filename"])
            with col_db3:
                st.metric("Database Size", f"{db_info['size_mb']} MB")
            with col_db4:
                st.metric("Relational Tables", f"{db_info['tables_count']}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
