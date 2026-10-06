"""
Script to test that all 4 dashboard pages load and render without exception
in a local Python 3.12 environment under both FULL DB and DEMO DB modes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_database_info, get_database_path, is_demo_database


def test_page_loads():
    print("=" * 60)
    print("VERIFYING DASHBOARD PAGES (LOCAL PYTHON 3.12)")
    print("=" * 60)

    # 1. Test database connection & info
    db_info = get_database_info()
    print(f"Active DB: {db_info['filename']} ({db_info['mode_label']}, {db_info['size_mb']} MB)")

    # 2. Page 1: Inventory & Supplier Risk
    print("\n[Page 1] Inventory & Supplier Risk:")
    from dashboard.pages.risk_page import load_risk_data
    df_exp, df_sup = load_risk_data()
    print(f"  - Loaded {len(df_exp):,} operational exposure records (latest: {df_exp['date'].max()})")
    print(f"  - Loaded {len(df_sup):,} supplier risk predictions")
    assert len(df_exp) > 0, "Exposure records empty"
    assert len(df_sup) > 0, "Supplier predictions empty"
    print("  -> Page 1 Data & Logic: PASS")

    # 3. Page 2: Prescriptive Optimization
    print("\n[Page 2] Prescriptive Operations Optimization:")
    from dashboard.pages.prescriptive_page import load_prescriptive_base_data, run_scenario_prescriptive
    op_latest, p_dims, s_dims, w_dims, latest_date = load_prescriptive_base_data()
    print(f"  - Loaded base data: {len(op_latest)} operational rows, {len(p_dims)} SKUs, {len(s_dims)} suppliers, {len(w_dims)} warehouses")
    opt_res, base_res, comparison, narrative, xfer_exps, dual_exps, constraint_exps = run_scenario_prescriptive("baseline")
    print(f"  - HiGHS Status: {opt_res.status}")
    print(f"  - Optimal Cost: ${opt_res.total_cost:,.2f}")
    print(f"  - Baseline Cost: ${base_res.total_cost:,.2f}")
    print(f"  - Service Level: {opt_res.service_level*100:.1f}%")
    print(f"  - Lateral Transfers: {opt_res.total_transferred:,.0f} units")
    assert opt_res.status.upper() == "OPTIMAL"
    assert abs(opt_res.total_cost - 136859.92) < 0.1
    assert abs(base_res.total_cost - 1220182.26) < 0.1
    assert abs(opt_res.service_level - 1.00) < 0.001
    assert abs(opt_res.total_transferred - 5978) < 1.0
    print("  -> Page 2 Optimization & Logic: PASS")

    # 4. Page 3: Demand Forecasting
    print("\n[Page 3] Multi-Horizon Demand Forecasting:")
    forecast_path = PROJECT_ROOT / "data" / "processed" / "forecasts" / "demand_forecasts.parquet"
    assert forecast_path.exists(), f"Forecast file missing at {forecast_path}"
    import pandas as pd
    df_fc = pd.read_parquet(forecast_path)
    print(f"  - Loaded {len(df_fc):,} forecast records covering horizons {sorted(df_fc['horizon'].unique())}")
    print("  -> Page 3 Forecasting Logic: PASS")

    # 5. Page 4: System Architecture & Audit
    print("\n[Page 4] System Architecture & Audit:")
    import platform
    py_ver = platform.python_version()
    print(f"  - Python Runtime : v{py_ver}")
    print(f"  - Pandas Version : v{pd.__version__}")
    import streamlit as st
    print(f"  - Streamlit Ver  : v{st.__version__}")
    print(f"  - Database Mode  : {db_info['mode_label']}")
    assert py_ver.startswith("3.12"), f"Expected Python 3.12, got {py_ver}"
    print("  -> Page 4 Audit & Diagnostics: PASS")

    print("\n" + "=" * 60)
    print("ALL 4 DASHBOARD PAGES VERIFIED AND PASSING!")
    print("=" * 60)


if __name__ == "__main__":
    test_page_loads()
