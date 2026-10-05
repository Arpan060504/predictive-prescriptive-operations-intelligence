"""
Main CLI entry point for the Predictive → Prescriptive Operations Intelligence Platform (PPOI).
Unified interface for data generation, quality auditing, database operations,
feature engineering, forecasting, risk modeling, scenario simulation, optimization,
dashboard launching, and automated testing.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_resolved_path

logger = get_logger("PPOI-CLI")


def main():
    parser = argparse.ArgumentParser(
        description="Predictive → Prescriptive Operations Intelligence Platform (PPOI) CLI"
    )
    parser.add_argument(
        "action",
        choices=[
            "generate-data",
            "validate-data",
            "build-db",
            "build-analytics",
            "build-features",
            "extract-features",
            "train-forecast",
            "train-risk-models",
            "score-risk",
            "run-risk-pipeline",
            "run-optimization",
            "generate-scenarios",
            "compare-policies",
            "sensitivity-analysis",
            "validate-optimization",
            "run-pipeline",
            "dashboard",
            "test"
        ],
        help="Action to execute within the intelligence pipeline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for data generation volume (default: 1.0, e.g. 0.2 for quick test)"
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run or launch in precomputed demo mode"
    )

    args = parser.parse_args()

    logger.info(f"Executing CLI action: '{args.action}' with seed={args.seed}, scale={args.scale}")

    if args.action == "generate-data":
        try:
            from src.data_generation.generator import run_data_generation
            run_data_generation(seed=args.seed, scale=args.scale)
        except ImportError as e:
            logger.error(f"Data generation module not yet implemented or error importing: {e}")
            sys.exit(1)

    elif args.action == "validate-data":
        try:
            from src.data_quality.auditor import run_quality_audit
            run_quality_audit()
        except ImportError as e:
            logger.error(f"Data quality audit module not yet implemented or error importing: {e}")
            sys.exit(1)

    elif args.action == "build-db":
        try:
            from src.database.builder import build_database
            build_database()
        except ImportError as e:
            logger.error(f"Database builder module not yet implemented or error importing: {e}")
            sys.exit(1)

    elif args.action == "build-analytics":
        try:
            from src.database.queries import get_db_connection
            from src.analytics.views import build_materialized_analytical_tables
            from src.analytics.validation import run_full_analytical_audit
            conn = get_db_connection()
            counts = build_materialized_analytical_tables(conn)
            audit_res = run_full_analytical_audit(conn)
            conn.close()
            print("\n" + "=" * 60)
            print("ANALYTICAL LAYER MATERIALIZATION COMPLETE")
            print("=" * 60)
            for tbl, cnt in counts.items():
                print(f"{tbl:35}: {cnt:,} rows")
            print("-" * 60)
            print(f"Analytical Audit Status          : {'ALL PASSED' if audit_res['all_passed'] else 'FAILED'}")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Analytical builder module failed or error: {e}")
            sys.exit(1)

    elif args.action in ["build-features", "extract-features"]:
        try:
            from src.features.feature_pipeline import run_feature_pipeline
            summary = run_feature_pipeline()
            print("\n" + "=" * 60)
            print("FEATURE ENGINEERING PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Demand Features     : {summary['demand_features']['rows']:,} rows × {summary['demand_features']['columns']} cols")
            print(f"Inventory Features  : {summary['inventory_risk_features']['rows']:,} rows × {summary['inventory_risk_features']['columns']} cols")
            print(f"Supplier Features   : {summary['supplier_risk_features']['rows']:,} rows × {summary['supplier_risk_features']['columns']} cols")
            if "leakage_validation" in summary:
                print(f"Leakage Validation  : {'ALL PASSED' if summary['leakage_validation']['all_leakage_free'] else 'FAILED'}")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action == "train-forecast":
        try:
            from src.forecasting.pipeline import run_forecast_pipeline
            run_forecast_pipeline()
        except Exception as e:
            logger.error(f"Forecasting pipeline failed or error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action in ["train-risk-models", "score-risk", "run-risk-pipeline"]:
        try:
            from src.risk_models.pipeline import run_risk_pipeline
            run_risk_pipeline(seed=args.seed)
        except Exception as e:
            logger.error(f"Risk models pipeline failed or error importing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action == "run-optimization":
        try:
            from src.optimization.engine import run_prescriptive_optimization
            run_prescriptive_optimization()
        except ImportError as e:
            logger.error(f"Optimization engine not yet implemented or error importing: {e}")
            sys.exit(1)

    elif args.action == "generate-scenarios":
        try:
            from src.simulation.simulator import run_scenario_generation
            run_scenario_generation()
        except Exception as e:
            logger.error(f"Scenario simulator failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action == "compare-policies":
        try:
            import sqlite3
            import pandas as pd
            from src.optimization.optimizer import NetworkOptimizer
            from src.optimization.baseline import BaselinePolicyEngine
            from src.decision_engine.policy_comparison import compare_policies
            from src.utils.config import get_resolved_path

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

            opt = NetworkOptimizer(op_latest, s_dims, w_dims, p_dims)
            opt_res = opt.solve("baseline")
            base = BaselinePolicyEngine(op_latest, s_dims, w_dims, p_dims)
            base_res = base.evaluate("baseline")
            comp = compare_policies(opt_res, base_res)

            print("\n" + "=" * 80)
            print("PRESCRIPTIVE VS BASELINE POLICY COMPARISON")
            print("=" * 80)
            print(f"Baseline Heuristic Total Cost : ${comp['baseline_total_cost']:>12,.2f} | Fill Rate: {comp['baseline_service_level']*100:.1f}%")
            print(f"Prescriptive Optimal Total Cost: ${comp['optimized_total_cost']:>12,.2f} | Fill Rate: {comp['optimized_service_level']*100:.1f}%")
            print(f"Modeled Landed Cost Reduction : -${comp['cost_difference']:>11,.2f} ({comp['cost_reduction_pct']:.1f}% lower)")
            print(f"Sourcing Concentration (HHI)  : Baseline {comp['baseline_hhi']} -> Optimized {comp['optimized_hhi']}")
            print(f"Lateral Transshipment Volume  : {comp['total_transfers_units']:>12,.0f} units")
            print("-" * 80)
            print("Cost Waterfall Breakdown:")
            print(comp["cost_waterfall"].to_string(index=False))
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Policy comparison failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action == "sensitivity-analysis":
        try:
            import sqlite3
            import pandas as pd
            from src.decision_engine.sensitivity import SensitivityAnalyzer
            from src.utils.config import get_resolved_path

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

            analyzer = SensitivityAnalyzer(op_latest, s_dims, w_dims, p_dims)
            sweeps = analyzer.run_all_sweeps()

            print("\n" + "=" * 80)
            print("PRESCRIPTIVE SENSITIVITY SWEEPS")
            print("=" * 80)
            for sweep_name, df_sweep in sweeps.items():
                print(f"\n--- {sweep_name.upper()} ---")
                print(df_sweep.to_string(index=False))
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.action == "validate-optimization":
        logger.info("Executing Phase 9 Prescriptive Optimization Validation Tests...")
        cmd = [sys.executable, "-m", "pytest", "tests/test_optimization.py", "-v"]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    elif args.action == "run-pipeline":
        logger.info("Executing end-to-end Operations Intelligence Pipeline...")
        # Step 1: Generate Data
        from src.data_generation.generator import run_data_generation
        run_data_generation(seed=args.seed, scale=args.scale)
        # Step 2: Validate Data Quality
        from src.data_quality.auditor import run_quality_audit
        run_quality_audit()
        # Step 3: Build Star-Schema SQLite Database
        from src.database.builder import build_database
        build_database()
        # Step 3b: Build Materialized Analytical Views & Tables
        from src.database.queries import get_db_connection
        from src.analytics.views import build_materialized_analytical_tables
        conn = get_db_connection()
        build_materialized_analytical_tables(conn)
        conn.close()
        # Step 4: Extract Features
        from src.feature_engineering.engineer import run_feature_engineering
        run_feature_engineering()
        # Step 5: Train Forecasting Hierarchy
        from src.forecasting.pipeline import run_forecast_pipeline
        run_forecast_pipeline()
        # Step 6: Train Risk Classifiers
        from src.risk_models.pipeline import run_risk_pipeline
        run_risk_pipeline()
        # Step 7: Run Prescriptive Optimization & Precompute Scenarios
        from src.optimization.engine import run_prescriptive_optimization
        run_prescriptive_optimization()
        from src.simulation.simulator import run_scenario_generation
        run_scenario_generation()
        logger.info("End-to-end Pipeline execution completed successfully.")

    elif args.action == "dashboard":
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        logger.info(f"Launching Streamlit dashboard from {app_path}...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        subprocess.run(cmd)

    elif args.action == "test":
        logger.info("Running automated test suite via pytest...")
        cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
