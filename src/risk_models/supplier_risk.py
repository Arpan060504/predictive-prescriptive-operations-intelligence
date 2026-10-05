"""
Supplier Delay Risk Modeling Engine for PPOI (Phase 8).
Implements candidate classifiers, calibration, feature importance,
and deterministic operational risk scoring for predicting P(supplier delivery delay).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.risk_models.metrics import evaluate_risk_classification
from src.risk_models.validation import (
    FORBIDDEN_SUPPLIER_COLUMNS,
    SUPPLIER_FEATURE_WHITELIST,
    assert_risk_feature_whitelist,
)
from src.utils.logger import get_logger

logger = get_logger("SupplierRisk")


class HistoricalSupplierBaseline(BaseEstimator, ClassifierMixin):
    """
    Empirical historical supplier performance baseline.
    Estimates delay probability based on historical unreliability (1 - hist_on_time_rate)
    or static prior baseline_reliability during cold start.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.global_delay_rate_ = 0.49

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is not None:
            self.global_delay_rate_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "hist_on_time_rate" in X.columns:
            on_time = X["hist_on_time_rate"].fillna(0.50).values
            is_cold = X["is_cold_start"].values if "is_cold_start" in X.columns else np.zeros(len(X))
            base_rel = X["baseline_reliability"].values if "baseline_reliability" in X.columns else np.full(len(X), 0.50)
            
            # Use baseline reliability if cold start
            effective_on_time = np.where(is_cold == 1, base_rel, on_time)
            prob_1 = np.clip(1.0 - effective_on_time, 0.05, 0.95)
        else:
            prob_1 = np.full(len(X), self.global_delay_rate_)
            
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def build_supplier_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Creates a ColumnTransformer for supplier features fitted strictly on training data."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


class SupplierRiskClassifier:
    """
    Unified interface for candidate supplier delay risk models (Baseline, Logistic, RF, XGBoost).
    """
    def __init__(
        self,
        model_family: str = "xgboost",
        random_state: int = 42,
        class_weight_ratio: Optional[float] = None,
        **kwargs
    ):
        self.model_family = model_family.lower()
        self.random_state = random_state
        self.class_weight_ratio = class_weight_ratio
        self.kwargs = kwargs
        self.model = None
        self.preprocessor = None
        self.calibrator = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting Supplier Risk Classifier: %s (samples=%d, features=%d)...",
                    self.model_family, len(X), X.shape[1])
        
        self.categorical_features = [c for c in ["supplier_id", "supplier_tier", "warehouse_id"] if c in X.columns]
        self.numeric_features = [c for c in X.columns if c not in self.categorical_features]
        self.feature_names = list(X.columns)

        if self.model_family == "baseline":
            self.model = HistoricalSupplierBaseline()
            self.model.fit(X, y)
            return self

        self.preprocessor = build_supplier_preprocessor(self.numeric_features, self.categorical_features)
        X_trans = self.preprocessor.fit_transform(X)

        if self.model_family == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                C=self.kwargs.get("C", 1.0),
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_family in ["random_forest", "rf"]:
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 6),
                min_samples_leaf=self.kwargs.get("min_samples_leaf", 10),
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_family in ["xgboost", "xgb"]:
            pos_rate = float(y.mean())
            spw = (1.0 - pos_rate) / max(pos_rate, 1e-4) if self.class_weight_ratio is None else self.class_weight_ratio
            self.model = xgb.XGBClassifier(
                n_estimators=self.kwargs.get("n_estimators", 80),
                max_depth=self.kwargs.get("max_depth", 4),
                learning_rate=self.kwargs.get("learning_rate", 0.06),
                subsample=self.kwargs.get("subsample", 0.8),
                colsample_bytree=self.kwargs.get("colsample_bytree", 0.8),
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown supplier model_family: {self.model_family}")

        self.model.fit(X_trans, y)
        logger.info("Fitting complete for %s.", self.model_family)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_family == "baseline":
            return self.model.predict_proba(X)
        
        X_trans = self.preprocessor.transform(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_trans)
        return self.model.predict_proba(X_trans)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid"):
        """Calibrates strictly on Validation set. Never on Test set."""
        if self.model_family == "baseline":
            logger.info("Skipping calibration for baseline.")
            return self
            
        logger.info("Calibrating supplier model %s on validation set (%d samples, method=%s)...",
                    self.model_family, len(X_val), method)
        X_val_trans = self.preprocessor.transform(X_val)
        
        calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv="prefit"
        )
        calibrator.fit(X_val_trans, y_val)
        self.calibrator = calibrator
        return self

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """Extracts top predictive feature importances."""
        if self.model_family == "baseline":
            return pd.DataFrame([{"feature": "hist_on_time_rate", "importance": 1.0, "family": "Historical Track Record"}])
            
        num_names = self.numeric_features
        cat_names = []
        if self.categorical_features:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        all_features = num_names + cat_names
        
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
            
        df_imp = pd.DataFrame({
            "feature": all_features,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        def get_family(name: str) -> str:
            if "hist_" in name or "recent_" in name or "trend" in name:
                return "Historical Track Record"
            elif "capacity" in name or "volume" in name or "active" in name:
                return "Capacity & Volume Pressure"
            elif "lead_time" in name or "delay" in name:
                return "Lead Time & Variance"
            elif "order_" in name or "quantity" in name or "cost" in name:
                return "Order Attributes"
            elif "supplier" in name or "tier" in name:
                return "Supplier Profile"
            return "Other"
            
        df_imp["family"] = df_imp["feature"].apply(get_family)
        return df_imp.head(top_n)


def compute_supplier_risk_score(
    df: pd.DataFrame,
    delay_prob: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes deterministic operational supplier risk score (0-100) and risk bands.
    
    Formula:
        Score = w_prob * (Prob * 100)
              + w_unrel * Historical_Unreliability_Score
              + w_drift * Performance_Drift_Score
              + w_cap * Capacity_Pressure_Score
              + w_sev * Delay_Severity_Score
    """
    if weights is None:
        weights = {
            "w_prob": 0.35,
            "w_unrel": 0.20,
            "w_drift": 0.15,
            "w_cap": 0.15,
            "w_sev": 0.15,
        }
    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}

    p_score = np.clip(delay_prob * 100.0, 0.0, 100.0)

    # 1. Historical unreliability score
    hist_ot = df["hist_on_time_rate"].fillna(0.5).values if "hist_on_time_rate" in df.columns else np.full(len(df), 0.5)
    unrel_score = np.clip((1.0 - hist_ot) * 100.0, 0.0, 100.0)

    # 2. Performance drift / recent deterioration
    recent_del = df["recent_delay_30d"].fillna(0.0).values if "recent_delay_30d" in df.columns else np.zeros(len(df))
    trend = df["delay_trend"].fillna(0.0).values if "delay_trend" in df.columns else np.zeros(len(df))
    drift_score = np.clip((recent_del / 3.0) * 50.0 + np.maximum(trend, 0.0) * 25.0, 0.0, 100.0)

    # 3. Capacity pressure score
    cap_press = df["supplier_capacity_pressure"].fillna(0.0).values if "supplier_capacity_pressure" in df.columns else np.zeros(len(df))
    cap_score = np.clip(cap_press * 100.0, 0.0, 100.0)

    # 4. Delay severity (P90 delay)
    p90 = df["hist_p90_delay"].fillna(0.0).values if "hist_p90_delay" in df.columns else np.zeros(len(df))
    sev_score = np.clip((p90 / 5.0) * 100.0, 0.0, 100.0)

    # Composite score
    composite_score = (
        w["w_prob"] * p_score +
        w["w_unrel"] * unrel_score +
        w["w_drift"] * drift_score +
        w["w_cap"] * cap_score +
        w["w_sev"] * sev_score
    )
    composite_score = np.clip(np.round(composite_score, 1), 0.0, 100.0)

    # Supplier Risk Bands
    # LOW: [0, 30), MEDIUM: [30, 60), HIGH: [60, 80), CRITICAL: [80, 100]
    bands = []
    for s in composite_score:
        if s < 30.0:
            bands.append("LOW")
        elif s < 60.0:
            bands.append("MEDIUM")
        elif s < 80.0:
            bands.append("HIGH")
        else:
            bands.append("CRITICAL")

    return composite_score, bands
