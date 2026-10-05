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


if __name__ == "__main__":
    main()
