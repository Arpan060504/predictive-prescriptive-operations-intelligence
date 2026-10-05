"""
Forecasting Evaluation Metrics Module for PPOI (Phase 7).
Provides mathematically stable, documented implementations of:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- Mean Error / Bias (Systematic Under/Over-forecasting)
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE = (1 / N) * sum(|y_true - y_pred|)
    Measures the average magnitude of absolute errors in physical units.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE = sqrt((1 / N) * sum((y_true - y_pred)^2))
    Penalizes large errors disproportionately, indicating tail forecast variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    Primary operations research metric. Avoids division-by-zero on intermittent demand
    by aggregating absolute volume across all SKU-warehouse observations before dividing.
    Returns 0.0 if sum(|y_true|) == 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        return 0.0
    num = float(np.sum(np.abs(y_true - y_pred)))
    return float(num / denom)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    sMAPE = (100% / N) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon) )
    Bounded percentage metric between 0% and 200%. Epsilon ensures numerical stability
    when both actual and prediction are zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    num = 2.0 * np.abs(y_true - y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(num / denom) * 100.0)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias = (1 / N) * sum(y_pred - y_true)
    Measures systematic tendency:
    - Positive bias (>0): Systematic over-forecasting (holding cost risk).
    - Negative bias (<0): Systematic under-forecasting (stockout / lost sales risk).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_pred - y_true))


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Computes all standard forecasting evaluation metrics in a single pass.
    Guarantees consistent rounding and type safety.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Valid mask in case of any NaNs
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]

    return {
        "mae": round(mean_absolute_error(y_t, y_p), 4),
        "rmse": round(root_mean_squared_error(y_t, y_p), 4),
        "wape": round(weighted_absolute_percentage_error(y_t, y_p), 4),
        "smape": round(symmetric_mean_absolute_percentage_error(y_t, y_p), 4),
        "bias": round(forecast_bias(y_t, y_p), 4),
        "n_samples": int(len(y_t)),
    }
