"""
Compatibility re-export layer for src.models.forecasting -> src.forecasting.
"""

from src.forecasting import (
    DEFAULT_FEATURE_WHITELIST,
    EmpiricalPredictionIntervals,
    MovingAverageForecaster,
    RidgeForecaster,
    SeasonalNaiveForecaster,
    XGBoostForecaster,
    assert_feature_whitelist,
    build_model_comparison_table,
    evaluate_forecast,
    extract_model_features,
    forecast_bias,
    mean_absolute_error,
    root_mean_squared_error,
    run_forecast_pipeline,
    select_best_model_per_horizon,
    symmetric_mean_absolute_percentage_error,
    verify_future_perturbation_invariance,
    weighted_absolute_percentage_error,
)

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
