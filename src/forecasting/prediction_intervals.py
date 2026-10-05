"""
Empirical Prediction Intervals Module for PPOI (Phase 7).
Constructs empirical prediction intervals around point forecasts using validation residuals.
Provides non-parametric quantiles, non-negative clipping, and test coverage evaluation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class EmpiricalPredictionIntervals:
    """
    Empirical Prediction Interval estimator based on historical validation residuals.
    
    Methodology:
    Let residuals be defined as e_i = y_i - y_hat_i on the out-of-sample validation split.
    For nominal coverage level (1 - alpha), e.g. 80% (alpha = 0.20):
        q_lower = quantile(e, alpha / 2)       e.g. 10th percentile
        q_upper = quantile(e, 1 - alpha / 2)   e.g. 90th percentile
    For future point forecast y_hat:
        Lower Bound = max(0.0, y_hat + q_lower)
        Upper Bound = max(Lower Bound, y_hat + q_upper)
    
    Assumptions & Limitations:
    1. Exchangeability: Assumes future forecast errors follow a similar distribution
       as validation period errors.
    2. Empirical heuristic: Provides empirical coverage benchmarks without claiming
       exact statistical finite-sample conformal validity.
    3. Truncation: Physical demand cannot be negative, so lower bounds are clipped at 0.0.
    """

    def __init__(self, nominal_coverage: float = 0.80):
        self.nominal_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage
        self.q_lower_: float = 0.0
        self.q_upper_: float = 0.0
        self.residual_mean_: float = 0.0
        self.residual_std_: float = 0.0
        self.n_residuals_: int = 0

    def fit(self, y_true_val: np.ndarray, y_pred_val: np.ndarray) -> "EmpiricalPredictionIntervals":
        """
        Estimates residual quantiles from validation actuals and predictions.
        """
        y_true = np.asarray(y_true_val, dtype=np.float64)
        y_pred = np.asarray(y_pred_val, dtype=np.float64)

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        residuals = y_true[mask] - y_pred[mask]

        if len(residuals) == 0:
            raise ValueError("No valid residuals provided to EmpiricalPredictionIntervals.")

        self.n_residuals_ = len(residuals)
        self.residual_mean_ = float(np.mean(residuals))
        self.residual_std_ = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        lower_pct = (self.alpha / 2.0) * 100.0
        upper_pct = (1.0 - self.alpha / 2.0) * 100.0

        self.q_lower_ = float(np.percentile(residuals, lower_pct))
        self.q_upper_ = float(np.percentile(residuals, upper_pct))

        return self

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (lower_bound, upper_bound) arrays for given point predictions.
        """
        preds = np.asarray(y_pred, dtype=np.float64)
        lower = np.maximum(0.0, preds + self.q_lower_)
        upper = np.maximum(lower, preds + self.q_upper_)
        return lower, upper

    def evaluate_coverage(
        self,
        y_true_test: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> Dict[str, float]:
        """
        Measures empirical coverage and interval sharpness on unseen test data.
        """
        y_t = np.asarray(y_true_test, dtype=np.float64)
        lb = np.asarray(lower_bound, dtype=np.float64)
        ub = np.asarray(upper_bound, dtype=np.float64)

        mask = ~np.isnan(y_t) & ~np.isnan(lb) & ~np.isnan(ub)
        y_t = y_t[mask]
        lb = lb[mask]
        ub = ub[mask]

        in_bounds = (y_t >= lb) & (y_t <= ub)
        empirical_coverage = float(np.mean(in_bounds))
        avg_width = float(np.mean(ub - lb))

        return {
            "nominal_coverage": round(self.nominal_coverage, 4),
            "empirical_coverage": round(empirical_coverage, 4),
            "average_interval_width": round(avg_width, 4),
            "q_lower": round(self.q_lower_, 4),
            "q_upper": round(self.q_upper_, 4),
            "n_samples": int(len(y_t)),
        }
