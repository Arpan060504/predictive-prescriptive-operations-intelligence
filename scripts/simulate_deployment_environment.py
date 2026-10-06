"""
Simulation script for Streamlit Community Cloud Deployment Environment.
Simulates the absence of database/operations.db and verifies that:
1. The resolver falls back cleanly to database/demo_operations.db.
2. All dashboard loading routines execute with zero errors.
3. Prescriptive optimization solves and generates valid policy comparisons.
4. Risk intelligence data loads properly.
"""

import os
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import (
    FULL_DB_PATH,
    DEMO_DB_PATH,
    get_database_info,
    get_database_path,
    is_demo_database,
)


def run_deployment_simulation():
    print("=" * 70)
    print("STREAMLIT CLOUD DEPLOYMENT SIMULATION")
    print("=" * 70)

    backup_path = PROJECT_ROOT / "database" / "operations.db.bak"
    full_db_existed = FULL_DB_PATH.exists()

    try:
        # Step 1: Hide full operations.db to mimic cloud clone
        if full_db_existed:
            print(f"Temporarily hiding {FULL_DB_PATH.name} -> {backup_path.name}...")
            shutil.move(str(FULL_DB_PATH), str(backup_path))

        print(f"Checking full DB exists : {FULL_DB_PATH.exists()} (Expected: False)")
        print(f"Checking demo DB exists : {DEMO_DB_PATH.exists()} (Expected: True)")
        assert not FULL_DB_PATH.exists(), "Full DB should not exist in simulation!"
        assert DEMO_DB_PATH.exists(), "Demo DB must exist in simulation!"

        # Step 2: Verify resolver telemetry
        info = get_database_info()
        print("\nResolved Database Telemetry:")
        for k, v in info.items():
            print(f"  {k:15}: {v}")
        assert info["is_demo"] is True, "Must resolve as demo database!"
        assert info["filename"] == "demo_operations.db", "Filename must be demo_operations.db!"

        # Step 3: Test Prescriptive Page Data Loading & Optimization
        print("\nTesting Prescriptive Optimization Engine with Demo Database...")
        from dashboard.pages.prescriptive_page import load_prescriptive_base_data, run_scenario_prescriptive
        op_latest, p_dims, s_dims, w_dims, latest_date = load_prescriptive_base_data()
        print(f"  Loaded operational records : {len(op_latest):,} (latest date: {latest_date})")
        print(f"  Loaded product dimensions   : {len(p_dims):,} SKUs")
        print(f"  Loaded supplier dimensions  : {len(s_dims):,} suppliers")
        print(f"  Loaded warehouse dimensions : {len(w_dims):,} warehouses")
        assert len(p_dims) == 60, f"Expected 60 products, got {len(p_dims)}"
        assert len(s_dims) == 8, f"Expected 8 suppliers, got {len(s_dims)}"
        assert len(w_dims) == 4, f"Expected 4 warehouses, got {len(w_dims)}"

        print("  Executing baseline scenario prescriptive optimization...")
        opt_res, base_res, comparison, narrative, xfer_exps, dual_exps, constraint_exps = run_scenario_prescriptive("baseline")
        print(f"  Optimizer Status           : {opt_res.status}")
        print(f"  Optimal Landed Cost        : ${opt_res.total_cost:,.2f}")
        print(f"  Baseline Landed Cost       : ${base_res.total_cost:,.2f}")
        print(f"  Fill Rate                  : {opt_res.service_level*100:.1f}%")
        print(f"  Lateral Transshipments     : {opt_res.total_transferred:,.0f} units")
        print(f"  Executive Narrative Length : {len(narrative)} chars")
        assert opt_res.status.upper() == "OPTIMAL", f"Optimizer failed with status: {opt_res.status}"
        assert abs(opt_res.total_cost - 136859.92) < 0.1, f"Expected 136,859.92, got {opt_res.total_cost}"
        assert abs(base_res.total_cost - 1220182.26) < 0.1, f"Expected 1,220,182.26, got {base_res.total_cost}"
        assert abs(opt_res.service_level - 1.00) < 0.001, f"Expected 100% fill rate, got {opt_res.service_level}"
        assert abs(opt_res.total_transferred - 5978) < 1.0, f"Expected 5,978 transfers, got {opt_res.total_transferred}"

        # Step 4: Test Risk Page Data Loading
        print("\nTesting Risk Intelligence Page Data Loading...")
        from dashboard.pages.risk_page import load_risk_data
        df_exp, df_sup = load_risk_data()
        print(f"  Loaded operational exposure records: {len(df_exp):,}")
        print(f"  Loaded supplier risk predictions   : {len(df_sup):,}")
        assert len(df_exp) > 0, "Operational exposure records must not be empty"
        assert len(df_sup) > 0, "Supplier risk records must not be empty"

        # Step 5: Test Optimization Unit Tests under Demo Mode
        print("\nTesting Prescriptive Optimization Unit Tests under Demo Mode...")
        import subprocess
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_optimization.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        print("  " + res.stdout.strip())
        assert res.returncode == 0, f"Optimization tests failed under demo mode:\n{res.stderr}"

        print("\n" + "=" * 70)
        print("DEPLOYMENT SIMULATION: ALL CHECKS PASSED (100% SUCCESS)")
        print("=" * 70)

    finally:
        # Step 6: Restore full operations.db
        if full_db_existed and backup_path.exists():
            print(f"\nRestoring {backup_path.name} -> {FULL_DB_PATH.name}...")
            shutil.move(str(backup_path), str(FULL_DB_PATH))
            print(f"Full DB restored: {FULL_DB_PATH.exists()} ({os.path.getsize(FULL_DB_PATH)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    run_deployment_simulation()
