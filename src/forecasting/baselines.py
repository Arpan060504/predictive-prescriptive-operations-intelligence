"""
Forecasting Baseline Models for PPOI (Phase 7).
Implements reproducible, leakage-free baselines:
1. SeasonalNaiveForecaster: 7-day seasonal cycle matching day-of-week phase.
2. MovingAverageForecaster: Historical moving average windows (7, 14, 28 days).
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Forecaster for weekly (7-day) seasonality.
    
    Mathematical Formulation:
    For a forecast origin at completion of day t and horizon h:
    The target date is t + h.
    The primary seasonal period is S = 7 days.
    The most recent observed date with the same seasonal phase (day-of-week) is:
        tau = (t + h) - 7 * ceil(h / 7)
    Notice that tau <= t, so y_tau was finalized at or before the forecast timestamp.
    In relative terms from t, the lookback lag is L = 7 * ceil(h / 7) - h >= 0:
        - For h = 1  (t+1): L = 6  -> forecast is y_{t-6} (same day-of-week 7 days prior)
        - For h = 7  (t+7): L = 0  -> forecast is y_t     (same day-of-week 7 days prior)
        - For h = 14 (t+14): L = 0 -> forecast is y_t
        - For h = 28 (t+28): L = 0 -> forecast is y_t
    """

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.series_history_: Dict[tuple, pd.Series] = {}
        self.global_fallback_: float = 0.0

    def fit(self, df_train: pd.DataFrame, target_col: Optional[str] = None) -> "SeasonalNaiveForecaster":
        """
        Stores historical series history up to training cutoff.
        """
        self.global_fallback_ = float(df_train["demand_requested"].median()) if "demand_requested" in df_train else 0.0
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        full_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generates seasonal-naive predictions for horizon h.
        df: Dataframe of rows at which forecasts are being generated (forecast origin dates).
        full_df: Full chronological dataframe used to resolve historical lags if df is a slice.
        """
        source_df = full_df if full_df is not None else df
        
        # Calculate lookback lag relative to day t
        k = int(np.ceil(horizon / self.season_length))
        lookback_lag = (self.season_length * k) - horizon

        # Sort and group by series keys
        sorted_source = source_df.sort_values(["product_id", "warehouse_id", "date"]).copy()
        
        if lookback_lag == 0:
            # y_t is demand_requested on day t
            sorted_source["_seasonal_naive_pred"] = sorted_source["demand_requested"].astype(float)
        else:
            # Shift within each series by lookback_lag
            sorted_source["_seasonal_naive_pred"] = (
                sorted_source.groupby(["product_id", "warehouse_id"])["demand_requested"]
                .shift(lookback_lag)
                .astype(float)
            )

        # Merge back onto df order
        lookup = sorted_source.set_index(["product_id", "warehouse_id", "date"])["_seasonal_naive_pred"]
        df_keys = pd.MultiIndex.from_frame(df[["product_id", "warehouse_id", "date"]])
        preds = lookup.reindex(df_keys).fillna(self.global_fallback_).values

        # Ensure non-negative demand
        return np.maximum(preds, 0.0)


class MovingAverageForecaster:
    """
    Moving Average Forecaster using historical windows (7, 14, 28 days).
    
    Mathematical Formulation:
    For forecast origin at completion of day t:
    Forecast = (1 / W) * sum_{k=0}^{W-1} y_{t-k}
    Uses strictly completed observations up to day t.
    """

    def __init__(self, window: int = 7):
        self.window = window
        self.global_fallback_: float = 0.0

    def fit(self, df_train: pd.DataFrame, target_col: Optional[str] = None) -> "MovingAverageForecaster":
        self.global_fallback_ = float(df_train["demand_requested"].median()) if "demand_requested" in df_train else 0.0
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        full_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        source_df = full_df if full_df is not None else df
        sorted_source = source_df.sort_values(["product_id", "warehouse_id", "date"]).copy()

        # Rolling mean of demand_requested including day t (since day t is completed)
        sorted_source["_ma_pred"] = (
            sorted_source.groupby(["product_id", "warehouse_id"])["demand_requested"]
            .transform(lambda s: s.rolling(window=self.window, min_periods=1).mean())
            .astype(float)
        )

        lookup = sorted_source.set_index(["product_id", "warehouse_id", "date"])["_ma_pred"]
        df_keys = pd.MultiIndex.from_frame(df[["product_id", "warehouse_id", "date"]])
        preds = lookup.reindex(df_keys).fillna(self.global_fallback_).values

        return np.maximum(preds, 0.0)
