"""
Model Selection Module for PPOI (Phase 7).
Compares candidate forecasting models on out-of-sample chronological validation data.
Selects optimal model family per horizon based on primary metric (WAPE) and locks configuration.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


def select_best_model_per_horizon(
    validation_results: List[Dict[str, Any]],
    primary_metric: str = "wape",
    secondary_metric: str = "mae"
) -> Dict[str, Dict[str, Any]]:
    """
    Selects the winning model for each forecast horizon based on validation metrics.
    
    Selection Rule:
    1. Filter candidates for horizon h.
    2. Sort ascending by primary_metric (default: WAPE).
    3. Tie-break using secondary_metric (default: MAE).
    4. Lock the selected configuration for final out-of-time test evaluation.
    """
    df = pd.DataFrame(validation_results)
    if df.empty:
        raise ValueError("Validation results list is empty; cannot select models.")

    selected_per_horizon = {}
    horizons = sorted(df["horizon"].unique())

    for h in horizons:
        sub = df[df["horizon"] == h].sort_values(by=[primary_metric, secondary_metric]).reset_index(drop=True)
        winner = sub.iloc[0].to_dict()
        selected_per_horizon[h] = {
            "horizon": h,
            "selected_model": winner["model"],
            "primary_metric": primary_metric,
            "primary_value": winner[primary_metric],
            "validation_metrics": {
                "mae": winner["mae"],
                "rmse": winner["rmse"],
                "wape": winner["wape"],
                "smape": winner["smape"],
                "bias": winner["bias"],
            },
            "selection_rationale": (
                f"Selected {winner['model']} for {h} with lowest validation {primary_metric.upper()} "
                f"({winner[primary_metric]:.4f}) vs competitors."
            ),
        }

    return selected_per_horizon


def build_model_comparison_table(validation_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats validation metrics into a standardized comparison DataFrame.
    """
    df = pd.DataFrame(validation_results)
    cols = ["model", "horizon", "mae", "rmse", "wape", "smape", "bias", "n_samples"]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].sort_values(["horizon", "wape"]).reset_index(drop=True)
