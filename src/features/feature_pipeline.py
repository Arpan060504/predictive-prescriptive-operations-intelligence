"""
Master Feature Pipeline Orchestration Module for PPOI (Phase 6).
Coordinates extraction, feature computation, leakage validation,
temporal partitioning, and artifact serialization to data/processed/features/.
"""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.database.queries import get_db_connection
from src.features.demand_features import compute_demand_features
from src.features.inventory_features import compute_inventory_risk_features
from src.features.supplier_features import compute_supplier_risk_features
from src.features.validation import (
    validate_demand_leakage,
    validate_inventory_leakage,
    validate_supplier_leakage,
    extract_base_demand_grid,
    extract_base_inventory_data,
    extract_base_purchase_orders
)
from src.utils.logger import get_logger

logger = get_logger("FeaturePipeline")

FEATURES_OUTPUT_DIR = Path("data/processed/features")


def get_temporal_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = "2025-03-31",
    val_end: str = "2025-06-30"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataset chronologically into Train, Validation, and Test sets:
    - Train:      Start to train_end (2024-01-01 to 2025-03-31: 15 months)
    - Validation: Day after train_end to val_end (2025-04-01 to 2025-06-30: 3 months)
    - Test:       Day after val_end to end (2025-07-01 to 2025-12-31: 6 months)
    
    Guarantees strict chronological ordering with zero temporal overlap.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    t_train_end = pd.to_datetime(train_end)
    t_val_end = pd.to_datetime(val_end)

    train_df = df[df[date_col] <= t_train_end].copy().reset_index(drop=True)
    val_df = df[(df[date_col] > t_train_end) & (df[date_col] <= t_val_end)].copy().reset_index(drop=True)
    test_df = df[df[date_col] > t_val_end].copy().reset_index(drop=True)

    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row count mismatch in temporal split!"
    assert train_df[date_col].max() < val_df[date_col].min(), "Leakage: Train overlaps Validation!"
    assert val_df[date_col].max() < test_df[date_col].min(), "Leakage: Validation overlaps Test!"

    logger.info(
        "Temporal Split for '%s': Train=%d rows (%s to %s), Val=%d rows (%s to %s), Test=%d rows (%s to %s)",
        date_col,
        len(train_df), train_df[date_col].min().strftime("%Y-%m-%d"), train_df[date_col].max().strftime("%Y-%m-%d"),
        len(val_df), val_df[date_col].min().strftime("%Y-%m-%d"), val_df[date_col].max().strftime("%Y-%m-%d"),
        len(test_df), test_df[date_col].min().strftime("%Y-%m-%d"), test_df[date_col].max().strftime("%Y-%m-%d")
    )

    return train_df, val_df, test_df


def run_feature_pipeline(
    output_dir: Path = FEATURES_OUTPUT_DIR,
    run_leakage_tests: bool = True
) -> Dict[str, Any]:
    """
    Runs the full Phase 6 Feature Engineering Pipeline:
    1. Computes features_demand (175,440 rows)
    2. Computes features_inventory_risk (175,440 rows)
    3. Computes features_supplier_risk (11,665 rows)
    4. Validates zero leakage across all three domains
    5. Saves artifacts in Parquet and CSV format
    6. Returns execution metadata summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    summary: Dict[str, Any] = {}

    try:
        # 1. Demand Features
        logger.info("--- Step 1: Demand Forecasting Feature Engineering ---")
        df_demand = compute_demand_features(conn)
        demand_parquet = output_dir / "demand_features.parquet"
        demand_csv = output_dir / "demand_features.csv"
        df_demand.to_parquet(demand_parquet, index=False)
        df_demand.to_csv(demand_csv, index=False)
        logger.info("Saved demand features to %s and %s", demand_parquet, demand_csv)
        summary["demand_features"] = {
            "rows": len(df_demand),
            "columns": len(df_demand.columns),
            "parquet_file": str(demand_parquet),
            "csv_file": str(demand_csv)
        }

        # 2. Inventory Risk Features
        logger.info("--- Step 2: Inventory Risk Feature Engineering ---")
        df_inventory = compute_inventory_risk_features(conn)
        inv_parquet = output_dir / "inventory_risk_features.parquet"
        inv_csv = output_dir / "inventory_risk_features.csv"
        df_inventory.to_parquet(inv_parquet, index=False)
        df_inventory.to_csv(inv_csv, index=False)
        logger.info("Saved inventory risk features to %s and %s", inv_parquet, inv_csv)
        summary["inventory_risk_features"] = {
            "rows": len(df_inventory),
            "columns": len(df_inventory.columns),
            "parquet_file": str(inv_parquet),
            "csv_file": str(inv_csv)
        }

        # 3. Supplier Risk Features
        logger.info("--- Step 3: Supplier Delay Risk Feature Engineering ---")
        df_supplier = compute_supplier_risk_features(conn)
        supp_parquet = output_dir / "supplier_risk_features.parquet"
        supp_csv = output_dir / "supplier_risk_features.csv"
        df_supplier.to_parquet(supp_parquet, index=False)
        df_supplier.to_csv(supp_csv, index=False)
        logger.info("Saved supplier risk features to %s and %s", supp_parquet, supp_csv)
        summary["supplier_risk_features"] = {
            "rows": len(df_supplier),
            "columns": len(df_supplier.columns),
            "parquet_file": str(supp_parquet),
            "csv_file": str(supp_csv)
        }

        # 4. Leakage Verification Tests
        if run_leakage_tests:
            logger.info("--- Step 4: Leakage Verification Tests ---")
            base_dem = extract_base_demand_grid(conn)
            dem_leakage = validate_demand_leakage(base_dem)
            
            base_inv = extract_base_inventory_data(conn)
            inv_leakage = validate_inventory_leakage(base_inv)

            base_po = extract_base_purchase_orders(conn)
            supp_leakage = validate_supplier_leakage(base_po)

            summary["leakage_validation"] = {
                "demand": dem_leakage,
                "inventory": inv_leakage,
                "supplier": supp_leakage,
                "all_leakage_free": (
                    dem_leakage["is_leakage_free"]
                    and inv_leakage["is_leakage_free"]
                    and supp_leakage["is_leakage_free"]
                )
            }
            logger.info("Leakage validation passed: %s", summary["leakage_validation"]["all_leakage_free"])

    finally:
        conn.close()

    # Save summary execution manifest
    manifest_path = output_dir / "pipeline_summary.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("Phase 6 Feature Pipeline executed successfully.")
    return summary


if __name__ == "__main__":
    run_feature_pipeline()
