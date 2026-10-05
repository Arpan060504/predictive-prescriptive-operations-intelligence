"""
PPOI Feature Engineering Package (Phase 6).
Provides production modules for leakage-safe demand forecasting,
inventory risk, supplier delay risk, and master pipeline orchestration.
"""

from src.features.demand_features import compute_demand_features
from src.features.inventory_features import compute_inventory_risk_features
from src.features.supplier_features import compute_supplier_risk_features
from src.features.feature_pipeline import run_feature_pipeline, get_temporal_split
from src.features.validation import (
    validate_demand_leakage,
    validate_inventory_leakage,
    validate_supplier_leakage,
    validate_feature_target_separation,
    validate_rolling_shift_invariance,
)

__all__ = [
    "compute_demand_features",
    "compute_inventory_risk_features",
    "compute_supplier_risk_features",
    "run_feature_pipeline",
    "get_temporal_split",
    "validate_demand_leakage",
    "validate_inventory_leakage",
    "validate_supplier_leakage",
    "validate_feature_target_separation",
    "validate_rolling_shift_invariance",
]
