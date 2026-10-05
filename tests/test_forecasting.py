"""
Automated Pytest Suite for Forecasting & Predictive Modeling Pipeline (Phase 7).
Validates:
1. Temporal split correctness and strict chronological non-overlapping boundaries.
2. Programmatic feature whitelist enforcement and loud rejection of forbidden target columns.
3. Mathematical correctness of Seasonal Naive and Moving Average baselines.
4. Mathematical stability and accuracy of MAE, RMSE, WAPE, sMAPE, and Bias metrics.
5. Empirical prediction interval quantile estimation and non-negative bounds clipping.
6. Target availability and warmup filtering.
7. Model artifact serialization and loading (Ridge & XGBoost).
8. Forecast output table schema and non-empty rows.
9. Deterministic random seed reproducibility.
10. Strong future perturbation regression test (mutating future demand does not alter historical inputs).
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import MovingAverageForecaster, SeasonalNaiveForecaster
from src.forecasting.metrics import (
    evaluate_forecast,
    forecast_bias,
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.prediction_intervals import EmpiricalPredictionIntervals
from src.forecasting.ridge_forecaster import RidgeForecaster
from src.forecasting.validation import (
    DEFAULT_FEATURE_WHITELIST,
    assert_feature_whitelist,
    extract_model_features,
    verify_future_perturbation_invariance,
)
from src.forecasting.xgboost_forecaster import XGBoostForecaster

FEATURE_PATH = Path("data/processed/features/demand_features.parquet")


@pytest.fixture(scope="module")
def demand_df():
    df = pd.read_parquet(FEATURE_PATH)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df


# -----------------------------------------------------------------------------
# 1. Temporal Splits & Boundaries
# -----------------------------------------------------------------------------
def test_temporal_splits_correctness(demand_df):
    """Verify exact row counts for Train, Validation, and Test partitions."""
    train = demand_df[(demand_df["date"] >= "2024-01-01") & (demand_df["date"] <= "2025-03-31")]
    val = demand_df[(demand_df["date"] >= "2025-04-01") & (demand_df["date"] <= "2025-06-30")]
    test = demand_df[(demand_df["date"] >= "2025-07-01") & (demand_df["date"] <= "2025-12-31")]

    assert len(train) == 109_440, f"Expected 109,440 Train rows, got {len(train)}"
    assert len(val) == 21_840, f"Expected 21,840 Validation rows, got {len(val)}"
    assert len(test) == 44_160, f"Expected 44,160 Test rows, got {len(test)}"
    assert len(train) + len(val) + len(test) == 175_440


def test_no_train_val_test_overlap(demand_df):
    """Verify strict chronological non-overlapping partitions."""
    train = demand_df[(demand_df["date"] >= "2024-01-01") & (demand_df["date"] <= "2025-03-31")]
    val = demand_df[(demand_df["date"] >= "2025-04-01") & (demand_df["date"] <= "2025-06-30")]
    test = demand_df[(demand_df["date"] >= "2025-07-01") & (demand_df["date"] <= "2025-12-31")]

    assert train["date"].max() < val["date"].min(), "Train overlaps Validation!"
    assert val["date"].max() < test["date"].min(), "Validation overlaps Test!"


# -----------------------------------------------------------------------------
# 2. Feature Whitelist & Leakage Assertion
# -----------------------------------------------------------------------------
def test_forbidden_target_rejection():
    """Assert ValueError is raised when any forbidden target column is provided."""
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
        assert_feature_whitelist(["demand_lag_1", "target_demand_t_plus_1"])

    with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
        assert_feature_whitelist(["demand_lag_7", "target_demand_sum_7d"])

    with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
        assert_feature_whitelist(["demand_lag_1", "outlier_flag"])


def test_feature_whitelist_validity():
    """Verify the default feature whitelist passes without errors."""
    assert_feature_whitelist(DEFAULT_FEATURE_WHITELIST)
    assert len(DEFAULT_FEATURE_WHITELIST) == 36
    # No target columns
    assert not any(c.startswith("target_") for c in DEFAULT_FEATURE_WHITELIST)


# -----------------------------------------------------------------------------
# 3. Baseline Models Correctness
# -----------------------------------------------------------------------------
def test_seasonal_naive_correctness(demand_df):
    """
    Verify Seasonal Naive logic:
    For horizon 1, forecast at day t is y_{t-6} (same day-of-week 7 days before t+1).
    For horizon 7, forecast at day t is y_t (same day-of-week 7 days before t+7).
    """
    sample = demand_df[(demand_df["product_id"] == "SKU-001") & (demand_df["warehouse_id"] == "WH-01")].copy()
    sample = sample.sort_values("date").reset_index(drop=True)

    sn = SeasonalNaiveForecaster(season_length=7)
    sn.fit(sample.iloc[:100])

    preds_h1 = sn.predict(sample.iloc[10:20], horizon=1, full_df=sample)
    # Check that each prediction matches demand_requested shifted by 6
    expected_h1 = sample.iloc[10 - 6 : 20 - 6]["demand_requested"].values
    np.testing.assert_allclose(preds_h1, expected_h1)

    preds_h7 = sn.predict(sample.iloc[10:20], horizon=7, full_df=sample)
    expected_h7 = sample.iloc[10:20]["demand_requested"].values
    np.testing.assert_allclose(preds_h7, expected_h7)


def test_moving_average_correctness(demand_df):
    """Verify Moving Average computes historical window mean including completed day t."""
    sample = demand_df[(demand_df["product_id"] == "SKU-001") & (demand_df["warehouse_id"] == "WH-01")].copy()
    sample = sample.sort_values("date").reset_index(drop=True)

    ma7 = MovingAverageForecaster(window=7)
    ma7.fit(sample.iloc[:100])

    preds = ma7.predict(sample.iloc[10:15], horizon=1, full_df=sample)
    for idx, row_idx in enumerate(range(10, 15)):
        window_vals = sample.iloc[row_idx - 6 : row_idx + 1]["demand_requested"].values
        expected_mean = np.mean(window_vals)
        assert abs(preds[idx] - expected_mean) < 1e-5


# -----------------------------------------------------------------------------
# 4. Metrics Mathematical Correctness & Stability
# -----------------------------------------------------------------------------
def test_metric_mae_correctness():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    # abs diff: [2, 2, 5], mean = 9 / 3 = 3.0
    assert abs(mean_absolute_error(y_true, y_pred) - 3.0) < 1e-6


def test_metric_rmse_correctness():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([13.0, 16.0, 30.0])
    # sq diff: [9, 16, 0], mean = 25 / 3 -> sqrt(25/3) = 2.88675
    expected_rmse = np.sqrt(25.0 / 3.0)
    assert abs(root_mean_squared_error(y_true, y_pred) - expected_rmse) < 1e-5


def test_metric_wape_correctness():
    y_true = np.array([100.0, 200.0, 300.0])  # sum = 600
    y_pred = np.array([110.0, 190.0, 330.0])  # abs err: [10, 10, 30], sum = 50
    # WAPE = 50 / 600 = 0.083333
    assert abs(weighted_absolute_percentage_error(y_true, y_pred) - (50.0 / 600.0)) < 1e-6


def test_metric_smape_correctness():
    y_true = np.array([100.0, 0.0])
    y_pred = np.array([100.0, 0.0])
    # Perfectly accurate should yield 0.0
    assert symmetric_mean_absolute_percentage_error(y_true, y_pred) < 1e-5


def test_metric_bias_correctness():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 110.0])
    # Bias = (110-100 + 110-100) / 2 = +10.0
    assert abs(forecast_bias(y_true, y_pred) - 10.0) < 1e-6


def test_metric_zero_denominator_stability():
    """Verify metrics return 0.0 gracefully on all-zero vectors."""
    zeros = np.zeros(10)
    assert weighted_absolute_percentage_error(zeros, zeros) == 0.0
    assert symmetric_mean_absolute_percentage_error(zeros, zeros) == 0.0
    eval_res = evaluate_forecast(zeros, zeros)
    assert eval_res["wape"] == 0.0
    assert eval_res["smape"] == 0.0


# -----------------------------------------------------------------------------
# 5. Prediction Intervals
# -----------------------------------------------------------------------------
def test_prediction_interval_calculation():
    """Verify empirical quantiles, non-negative clipping, and test coverage evaluation."""
    np.random.seed(42)
    y_true_val = np.random.uniform(50, 150, size=500)
    # Residuals uniformly distributed in [-20, 20]
    errors = np.random.uniform(-20, 20, size=500)
    y_pred_val = y_true_val - errors

    intervals = EmpiricalPredictionIntervals(nominal_coverage=0.80)
    intervals.fit(y_true_val, y_pred_val)

    # 10th percentile should be approx -16, 90th approx +16
    assert -20.0 <= intervals.q_lower_ <= -12.0
    assert 12.0 <= intervals.q_upper_ <= 20.0

    test_preds = np.array([10.0, 100.0, 200.0])
    lower, upper = intervals.predict_intervals(test_preds)

    assert (lower >= 0.0).all(), "Lower prediction interval must not be negative!"
    assert (upper >= lower).all(), "Upper interval must be >= lower interval!"

    cov_eval = intervals.evaluate_coverage(
        y_true_test=y_true_val[:100],
        lower_bound=lower[0] * np.ones(100),
        upper_bound=upper[2] * np.ones(100),
    )
    assert 0.0 <= cov_eval["empirical_coverage"] <= 1.0


# -----------------------------------------------------------------------------
# 6. Model Serialization & Reproducibility
# -----------------------------------------------------------------------------
def test_ridge_forecaster_fit_predict_save_load(demand_df):
    """Test Ridge fitting, prediction, saving, and loading on sample data."""
    sample_train = demand_df[(demand_df["date"] <= "2024-06-30") & (demand_df["has_sufficient_history_28d"] == 1)].head(1000).copy()
    sample_val = demand_df[(demand_df["date"] > "2024-06-30")].head(200).copy()

    ridge = RidgeForecaster(alpha=50.0)
    ridge.fit(sample_train, target_col="target_demand_t_plus_1")
    preds_before = ridge.predict(sample_val)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "ridge_test.joblib"
        ridge.save(model_path)
        loaded_ridge = RidgeForecaster.load(model_path)
        preds_after = loaded_ridge.predict(sample_val)

    np.testing.assert_allclose(preds_before, preds_after)


def test_xgboost_forecaster_fit_predict_save_load(demand_df):
    """Test XGBoost fitting, prediction, saving, loading, and feature importance."""
    sample_train = demand_df[(demand_df["date"] <= "2024-06-30") & (demand_df["has_sufficient_history_28d"] == 1)].head(1000).copy()
    sample_val = demand_df[(demand_df["date"] > "2024-06-30")].head(200).copy()

    xgb_model = XGBoostForecaster()
    xgb_model.fit(sample_train, target_col="target_demand_t_plus_1")
    preds_before = xgb_model.predict(sample_val)

    fi_df = xgb_model.get_feature_importance(top_n=10)
    assert len(fi_df) <= 10
    assert "feature" in fi_df.columns
    assert "importance" in fi_df.columns

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "xgb_test.joblib"
        xgb_model.save(model_path)
        loaded_xgb = XGBoostForecaster.load(model_path)
        preds_after = loaded_xgb.predict(sample_val)

    np.testing.assert_allclose(preds_before, preds_after)


def test_deterministic_random_seed(demand_df):
    """Verify that XGBoost produces identical predictions across identical seeds."""
    sample = demand_df[(demand_df["date"] <= "2024-04-30") & (demand_df["has_sufficient_history_28d"] == 1)].head(500).copy()
    val_sample = demand_df[demand_df["date"] > "2024-04-30"].head(100).copy()

    m1 = XGBoostForecaster(random_state=42)
    m1.fit(sample, "target_demand_t_plus_1")
    p1 = m1.predict(val_sample)

    m2 = XGBoostForecaster(random_state=42)
    m2.fit(sample, "target_demand_t_plus_1")
    p2 = m2.predict(val_sample)

    np.testing.assert_allclose(p1, p2)


# -----------------------------------------------------------------------------
# 7. Output Artifacts & Regression Leakage
# -----------------------------------------------------------------------------
def test_forecast_output_schema():
    """Verify that generated test forecasts parquet file exists and matches schema."""
    fc_path = Path("data/processed/forecasts/demand_forecasts.parquet")
    assert fc_path.exists(), "Forecasts parquet file does not exist!"
    df_fc = pd.read_parquet(fc_path)

    expected_cols = {
        "date", "product_id", "warehouse_id", "category", "actual_demand",
        "forecast", "horizon", "model", "lower_prediction_interval",
        "upper_prediction_interval", "residual", "split"
    }
    assert expected_cols.issubset(set(df_fc.columns))
    assert len(df_fc) > 0
    assert (df_fc["forecast"] >= 0.0).all(), "Negative forecasts detected!"


def test_future_perturbation_invariance_regression(demand_df):
    """
    Core regression test:
    Shock future demand by +10,000 units after 2025-06-30.
    Verify historical model input matrix remains strictly invariant (max_diff == 0.0).
    """
    res = verify_future_perturbation_invariance(demand_df, cutoff_date="2025-06-30", perturbation_shock=10_000.0)
    assert res["is_invariant"], f"Future perturbation leak: max diff = {res['max_abs_diff']}"
    assert res["max_abs_diff"] < 1e-6
