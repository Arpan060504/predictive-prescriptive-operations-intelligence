"""
Inventory Stockout Risk Modeling Engine for PPOI (Phase 8).
Implements candidate risk classifiers, calibration, feature importance,
and deterministic operational risk scoring for predicting P(stockout within 7 days).
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
from src.risk_models.validation import INVENTORY_FEATURE_WHITELIST, assert_risk_feature_whitelist, FORBIDDEN_INVENTORY_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("InventoryRisk")


class NaiveInventoryRiskBaseline(BaseEstimator, ClassifierMixin):
    """
    Deterministic rule-based inventory buffer baseline.
    Estimates stockout probability as a sigmoid function of days-of-supply deficiency
    and safety-stock deficit.
    """
    def __init__(self, dos_target: float = 7.0, scale: float = 2.0):
        self.dos_target = dos_target
        self.scale = scale
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dos = X["inventory_days_of_supply"].values if "inventory_days_of_supply" in X else np.full(len(X), 14.0)
        # Sigmoid: high when DOS is low (< 7 days)
        z = (self.dos_target - dos) / max(self.scale, 0.1)
        prob_1 = 1.0 / (1.0 + np.exp(-z))
        
        # Adjust for safety stock gap if present
        if "safety_stock_gap" in X.columns:
            ss_gap = X["safety_stock_gap"].values
            gap_penalty = np.where(ss_gap < 0, 0.15, 0.0)
            prob_1 = np.clip(prob_1 + gap_penalty, 0.01, 0.99)
            
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer fitted strictly on training data."""
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


class InventoryRiskClassifier:
    """
    Unified interface for candidate inventory risk models (Logistic, Random Forest, XGBoost).
    Handles feature preprocessing, class imbalance adjustment, calibration, and prediction.
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
        logger.info("Fitting Inventory Risk Classifier: %s (samples=%d, features=%d)...",
                    self.model_family, len(X), X.shape[1])
        
        # Categorize columns
        self.categorical_features = [c for c in ["category", "criticality", "warehouse_id"] if c in X.columns]
        self.numeric_features = [c for c in X.columns if c not in self.categorical_features]
        self.feature_names = list(X.columns)

        if self.model_family == "naive":
            self.model = NaiveInventoryRiskBaseline()
            self.model.fit(X, y)
            return self

        # Build preprocessor
        self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
        X_trans = self.preprocessor.fit_transform(X)

        pos_rate = float(y.mean())
        default_scale_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-4) if self.class_weight_ratio is None else self.class_weight_ratio

        if self.model_family == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                C=self.kwargs.get("C", 0.5),
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_family in ["random_forest", "rf"]:
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 120),
                max_depth=self.kwargs.get("max_depth", 8),
                min_samples_leaf=self.kwargs.get("min_samples_leaf", 15),
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_family in ["xgboost", "xgb"]:
            self.model = xgb.XGBClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 5),
                learning_rate=self.kwargs.get("learning_rate", 0.08),
                subsample=self.kwargs.get("subsample", 0.8),
                colsample_bytree=self.kwargs.get("colsample_bytree", 0.8),
                scale_pos_weight=default_scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_family: {self.model_family}")

        self.model.fit(X_trans, y)
        logger.info("Fitting complete for %s.", self.model_family)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_family == "naive":
            return self.model.predict_proba(X)
        
        X_trans = self.preprocessor.transform(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_trans)
        return self.model.predict_proba(X_trans)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid"):
        """
        Calibrates model output probabilities strictly on Validation set.
        Never calibrates on Test set.
        """
        if self.model_family == "naive":
            logger.info("Skipping calibration for naive rule baseline.")
            return self
            
        logger.info("Calibrating %s on validation set (%d samples, method=%s)...",
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
        if self.model_family == "naive":
            return pd.DataFrame([{"feature": "inventory_days_of_supply", "importance": 1.0, "family": "Buffer"}])
            
        # Get encoded column names
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
        
        # Categorize family
        def get_family(name: str) -> str:
            if "supply" in name or "inventory" in name or "buffer" in name or "stock" in name:
                return "Inventory Buffer"
            elif "demand" in name or "forecast" in name:
                return "Demand & Forecast"
            elif "stockout" in name or "lost_sales" in name:
                return "Historical Stockout"
            elif "warehouse" in name:
                return "Warehouse Utilization"
            elif "category" in name or "criticality" in name:
                return "Product Dimension"
            return "Other"
            
        df_imp["family"] = df_imp["feature"].apply(get_family)
        return df_imp.head(top_n)


def compute_inventory_risk_score(
    df: pd.DataFrame,
    stockout_prob: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes deterministic operational inventory risk score (0-100) and risk bands.
    
    Formula:
        Score = w_prob * (Prob * 100)
              + w_dos * DOS_Deficit_Score
              + w_ss * Safety_Stock_Deficit_Score
              + w_dem * Forecast_Demand_Exposure_Score
              + w_hist * Historical_Stockout_Score
    """
    if weights is None:
        weights = {
            "w_prob": 0.35,
            "w_dos": 0.20,
            "w_ss": 0.15,
            "w_dem": 0.15,
            "w_hist": 0.15,
        }
    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}

    p_score = np.clip(stockout_prob * 100.0, 0.0, 100.0)

    # 1. DOS deficiency: 0 if DOS >= 14, 100 if DOS <= 0
    dos = pd.Series(df.get("inventory_days_of_supply", 14.0)).fillna(14.0).values
    dos_score = np.clip((1.0 - (dos / 14.0)) * 100.0, 0.0, 100.0)

    # 2. Safety stock deficit: 100 if inventory == 0, 0 if inventory >= safety stock
    inv = pd.Series(df.get("ending_inventory_lag1", 0.0)).fillna(df.get("beginning_inventory", 0.0)).fillna(0.0).values
    ss_target = pd.Series(df.get("safety_stock_target", 1.0)).fillna(1.0).replace(0, 1.0).values
    ss_ratio = np.clip(inv / ss_target, 0.0, 2.0)
    ss_score = np.clip((1.0 - (ss_ratio / 1.0)) * 100.0, 0.0, 100.0)

    # 3. Forecast demand exposure: ratio of 7-day forecast demand to current buffer
    fc_dem = pd.Series(df.get("forecast_demand_7d", df.get("demand_mean_7", 10.0) * 7.0)).fillna(70.0).values
    dem_coverage = np.clip(fc_dem / np.maximum(inv, 1.0), 0.0, 2.0)
    dem_score = np.clip((dem_coverage / 1.5) * 100.0, 0.0, 100.0)

    # 4. Historical stockout rate
    hist_rate = pd.Series(df.get("stockout_rate_28d", 0.0)).fillna(0.0).values
    hist_score = np.clip(hist_rate * 100.0, 0.0, 100.0)

    # Composite score
    composite_score = (
        w["w_prob"] * p_score +
        w["w_dos"] * dos_score +
        w["w_ss"] * ss_score +
        w["w_dem"] * dem_score +
        w["w_hist"] * hist_score
    )
    composite_score = np.nan_to_num(composite_score, nan=0.0)
    composite_score = np.clip(np.round(composite_score, 1), 0.0, 100.0)

    # Risk Band Assignment
    # LOW: [0, 25), MEDIUM: [25, 60), HIGH: [60, 80), CRITICAL: [80, 100]
    bands = []
    for s in composite_score:
        if s < 25.0:
            bands.append("LOW")
        elif s < 60.0:
            bands.append("MEDIUM")
        elif s < 80.0:
            bands.append("HIGH")
        else:
            bands.append("CRITICAL")

    return composite_score, bands
