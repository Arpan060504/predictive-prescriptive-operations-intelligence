"""
PPOI Feature Engineering Module alias for src.features.
Ensures backwards and forwards compatibility across namespaces.
"""

from src.features.demand_features import compute_demand_features
from src.features.inventory_features import compute_inventory_risk_features
from src.features.supplier_features import compute_supplier_risk_features
from src.features.feature_pipeline import run_feature_pipeline, get_temporal_split

__all__ = [
    "compute_demand_features",
    "compute_inventory_risk_features",
    "compute_supplier_risk_features",
    "run_feature_pipeline",
    "get_temporal_split",
]
