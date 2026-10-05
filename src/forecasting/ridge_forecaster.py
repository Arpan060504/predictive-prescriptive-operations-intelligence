"""
Ridge Regression Forecaster Module for PPOI (Phase 7).
Implements a leakage-safe scikit-learn Pipeline with:
- StandardScaler and SimpleImputer on numerical whitelist features.
- OneHotEncoder on categorical whitelist features (fitted ONLY on training data).
- Hyperparameter alpha tuning on chronological validation data.
- Non-negative prediction clipping.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.forecasting.metrics import weighted_absolute_percentage_error
from src.forecasting.validation import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    assert_feature_whitelist,
)
from src.utils.logger import get_logger

logger = get_logger("RidgeForecaster")


class RidgeForecaster:
    """
    Ridge Regression demand forecaster with scikit-learn preprocessing.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
        self.categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
        self.random_state = random_state
        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_: List[str] = self.numeric_features + self.categorical_features

        # Assert whitelist immediately upon initialization
        assert_feature_whitelist(self.feature_names_)

    def _build_pipeline(self, alpha: float) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    self.numeric_features,
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha, random_state=self.random_state)),
        ])

    def fit(self, df_train: pd.DataFrame, target_col: str) -> "RidgeForecaster":
        """
        Fits preprocessing and Ridge model strictly on training data.
        """
        assert_feature_whitelist(self.feature_names_)
        if target_col not in df_train.columns:
            raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")

        # Respect warmup flag if present
        if "has_sufficient_history_28d" in df_train.columns:
            train_subset = df_train[df_train["has_sufficient_history_28d"] == 1].copy()
        else:
            train_subset = df_train.copy()

        # Remove rows where target is NaN
        valid_mask = train_subset[target_col].notna()
        train_subset = train_subset[valid_mask]

        X_train = train_subset[self.feature_names_]
        y_train = train_subset[target_col].values

        self.pipeline_ = self._build_pipeline(self.alpha)
        self.pipeline_.fit(X_train, y_train)
        logger.debug("Ridge model fitted with alpha=%.2f on %d samples.", self.alpha, len(y_train))
        return self

    def tune_alpha_on_validation(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_col: str,
        candidate_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 500.0, 1000.0),
    ) -> float:
        """
        Evaluates candidate alpha values on validation WAPE and locks the best alpha.
        """
        assert_feature_whitelist(self.feature_names_)
        
        # Train data
        if "has_sufficient_history_28d" in df_train.columns:
            train_subset = df_train[df_train["has_sufficient_history_28d"] == 1].copy()
        else:
            train_subset = df_train.copy()
        train_subset = train_subset[train_subset[target_col].notna()]
        X_tr = train_subset[self.feature_names_]
        y_tr = train_subset[target_col].values

        # Validation data
        val_subset = df_val[df_val[target_col].notna()].copy()
        X_v = val_subset[self.feature_names_]
        y_v = val_subset[target_col].values

        best_alpha = self.alpha
        best_wape = float("inf")

        for alpha_cand in candidate_alphas:
            pipe = self._build_pipeline(alpha_cand)
            pipe.fit(X_tr, y_tr)
            preds = np.maximum(pipe.predict(X_v), 0.0)
            wape = weighted_absolute_percentage_error(y_v, preds)
            logger.debug("Ridge alpha=%.1f -> Validation WAPE: %.4f", alpha_cand, wape)
            if wape < best_wape:
                best_wape = wape
                best_alpha = alpha_cand

        self.alpha = best_alpha
        # Refit pipeline with best alpha
        self.pipeline_ = self._build_pipeline(self.alpha)
        self.pipeline_.fit(X_tr, y_tr)
        logger.info("Selected optimal Ridge alpha=%.1f (Val WAPE=%.4f) for target '%s'", self.alpha, best_wape, target_col)
        return self.alpha

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for feature records and clips at 0.0 (non-negative demand).
        """
        if self.pipeline_ is None:
            raise RuntimeError("RidgeForecaster must be fitted before predict() is called.")
        assert_feature_whitelist(self.feature_names_)
        X = df[self.feature_names_]
        raw_preds = self.pipeline_.predict(X)
        return np.maximum(raw_preds, 0.0)

    def save(self, filepath: Path) -> None:
        """Serializes the fitted pipeline and configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline_,
            "alpha": self.alpha,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "random_state": self.random_state,
        }, filepath)
        logger.info("Saved RidgeForecaster to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "RidgeForecaster":
        """Loads a fitted RidgeForecaster from disk."""
        data = joblib.load(filepath)
        forecaster = cls(
            alpha=data["alpha"],
            numeric_features=data["numeric_features"],
            categorical_features=data["categorical_features"],
            random_state=data["random_state"],
        )
        forecaster.pipeline_ = data["pipeline"]
        return forecaster
