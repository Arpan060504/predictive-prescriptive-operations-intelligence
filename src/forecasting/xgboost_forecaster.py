"""
XGBoost Regressor Forecaster Module for PPOI (Phase 7).
Implements a conservative, leakage-safe XGBoost forecasting pipeline with:
- OneHotEncoder on categorical whitelist features (fitted ONLY on training data).
- Controlled hyperparameter evaluation on chronological validation data.
- Extraction of model-derived feature importance (gain/weight).
- Non-negative prediction clipping.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from src.forecasting.metrics import weighted_absolute_percentage_error
from src.forecasting.validation import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    assert_feature_whitelist,
)
from src.utils.logger import get_logger

logger = get_logger("XGBoostForecaster")

# Conservative candidate parameter configurations
XGBOOST_CANDIDATE_CONFIGS = [
    {
        "config_name": "balanced_default",
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    },
    {
        "config_name": "conservative_regularized",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
    },
    {
        "config_name": "deep_expressive",
        "n_estimators": 120,
        "max_depth": 6,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
    },
]


class XGBoostForecaster:
    """
    XGBoost Demand Forecaster with leakage-safe training and feature importance.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.params = params or XGBOOST_CANDIDATE_CONFIGS[0].copy()
        self.numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
        self.categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
        self.random_state = random_state
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.model_: Optional[xgb.XGBRegressor] = None
        self.feature_names_: List[str] = self.numeric_features + self.categorical_features
        self.encoded_feature_names_: List[str] = []

        assert_feature_whitelist(self.feature_names_)

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

    def fit(self, df_train: pd.DataFrame, target_col: str) -> "XGBoostForecaster":
        """
        Fits preprocessing and XGBoost model strictly on training data.
        """
        assert_feature_whitelist(self.feature_names_)
        if target_col not in df_train.columns:
            raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")

        # Respect warmup flag
        if "has_sufficient_history_28d" in df_train.columns:
            train_subset = df_train[df_train["has_sufficient_history_28d"] == 1].copy()
        else:
            train_subset = df_train.copy()

        train_subset = train_subset[train_subset[target_col].notna()]
        X_raw = train_subset[self.feature_names_]
        y_train = train_subset[target_col].values

        self.preprocessor_ = self._build_preprocessor()
        X_train_enc = self.preprocessor_.fit_transform(X_raw)

        # Store encoded column names for feature importance reporting
        cat_encoder = self.preprocessor_.named_transformers_["cat"]
        cat_enc_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.encoded_feature_names_ = self.numeric_features + cat_enc_names

        # Build XGBRegressor
        model_params = {k: v for k, v in self.params.items() if k != "config_name"}
        self.model_ = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            **model_params,
        )
        self.model_.fit(X_train_enc, y_train)
        logger.debug("XGBoost model fitted with %s on %d samples.", self.params.get("config_name"), len(y_train))
        return self

    def tune_on_validation(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_col: str,
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate hyperparameter configurations on validation WAPE and locks the best.
        """
        configs = candidate_configs or XGBOOST_CANDIDATE_CONFIGS
        assert_feature_whitelist(self.feature_names_)

        # Prepare train data
        if "has_sufficient_history_28d" in df_train.columns:
            train_subset = df_train[df_train["has_sufficient_history_28d"] == 1].copy()
        else:
            train_subset = df_train.copy()
        train_subset = train_subset[train_subset[target_col].notna()]
        X_tr_raw = train_subset[self.feature_names_]
        y_tr = train_subset[target_col].values

        preprocessor = self._build_preprocessor()
        X_tr_enc = preprocessor.fit_transform(X_tr_raw)

        # Prepare validation data
        val_subset = df_val[df_val[target_col].notna()].copy()
        X_v_raw = val_subset[self.feature_names_]
        y_v = val_subset[target_col].values
        X_v_enc = preprocessor.transform(X_v_raw)

        best_config = configs[0]
        best_wape = float("inf")

        for cfg in configs:
            m_params = {k: v for k, v in cfg.items() if k != "config_name"}
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=-1,
                **m_params,
            )
            model.fit(X_tr_enc, y_tr)
            preds = np.maximum(model.predict(X_v_enc), 0.0)
            wape = weighted_absolute_percentage_error(y_v, preds)
            logger.debug("XGBoost config '%s' -> Val WAPE: %.4f", cfg.get("config_name"), wape)
            if wape < best_wape:
                best_wape = wape
                best_config = cfg

        self.params = best_config.copy()
        self.preprocessor_ = preprocessor
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_enc_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.encoded_feature_names_ = self.numeric_features + cat_enc_names

        # Refit final model on training set with locked best configuration
        final_params = {k: v for k, v in self.params.items() if k != "config_name"}
        self.model_ = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            **final_params,
        )
        self.model_.fit(X_tr_enc, y_tr)

        logger.info(
            "Selected optimal XGBoost config '%s' (Val WAPE=%.4f) for target '%s'",
            self.params.get("config_name"),
            best_wape,
            target_col,
        )
        return self.params

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for feature records and clips at 0.0 (non-negative demand).
        """
        if self.model_ is None or self.preprocessor_ is None:
            raise RuntimeError("XGBoostForecaster must be fitted before predict() is called.")
        assert_feature_whitelist(self.feature_names_)
        X_raw = df[self.feature_names_]
        X_enc = self.preprocessor_.transform(X_raw)
        raw_preds = self.model_.predict(X_enc)
        return np.maximum(raw_preds, 0.0)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Returns a DataFrame of top model-derived feature importances (gain/weight).
        Explicitly indicates statistical association, NOT causal importance.
        """
        if self.model_ is None or not self.encoded_feature_names_:
            raise RuntimeError("Model must be fitted before extracting feature importance.")

        importances = self.model_.feature_importances_
        fi_df = pd.DataFrame({
            "feature": self.encoded_feature_names_,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return fi_df.head(top_n)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model_,
            "preprocessor": self.preprocessor_,
            "params": self.params,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "encoded_feature_names": self.encoded_feature_names_,
            "random_state": self.random_state,
        }, filepath)
        logger.info("Saved XGBoostForecaster to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "XGBoostForecaster":
        data = joblib.load(filepath)
        forecaster = cls(
            params=data["params"],
            numeric_features=data["numeric_features"],
            categorical_features=data["categorical_features"],
            random_state=data["random_state"],
        )
        forecaster.model_ = data["model"]
        forecaster.preprocessor_ = data["preprocessor"]
        forecaster.encoded_feature_names_ = data["encoded_feature_names"]
        return forecaster
