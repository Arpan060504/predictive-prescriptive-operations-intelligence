"""
Forecasting Package for PPOI (Phase 7).
Provides predictive demand forecasting models, metrics, prediction intervals,
model selection, and pipeline orchestration.
"""

from src.forecasting.baselines import MovingAverageForecaster, SeasonalNaiveForecaster
from src.forecasting.metrics import (
    evaluate_forecast,
    forecast_bias,
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.model_selection import (
    build_model_comparison_table,
    select_best_model_per_horizon,
)
from src.forecasting.pipeline import run_forecast_pipeline
from src.forecasting.prediction_intervals import EmpiricalPredictionIntervals
from src.forecasting.ridge_forecaster import RidgeForecaster
from src.forecasting.validation import (
    DEFAULT_FEATURE_WHITELIST,
    assert_feature_whitelist,
    extract_model_features,
    verify_future_perturbation_invariance,
)
from src.forecasting.xgboost_forecaster import XGBoostForecaster

__all__ = [
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "RidgeForecaster",
    "XGBoostForecaster",
    "EmpiricalPredictionIntervals",
    "mean_absolute_error",
    "root_mean_squared_error",
    "weighted_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "forecast_bias",
    "evaluate_forecast",
    "assert_feature_whitelist",
    "extract_model_features",
    "verify_future_perturbation_invariance",
    "select_best_model_per_horizon",
    "build_model_comparison_table",
    "run_forecast_pipeline",
    "DEFAULT_FEATURE_WHITELIST",
]
