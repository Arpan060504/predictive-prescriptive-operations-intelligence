"""
PPOI Risk Models Package (Phase 8).
Predictive risk modeling, probability calibration, and operational exposure intelligence.
"""

from src.risk_models.metrics import (
    calculate_top_k_capture,
    compute_brier_skill_score,
    evaluate_risk_classification,
)
from src.risk_models.validation import (
    INVENTORY_FEATURE_WHITELIST,
    SUPPLIER_FEATURE_WHITELIST,
    assert_risk_feature_whitelist,
    verify_future_perturbation_invariance,
)
from src.risk_models.inventory_risk import (
    InventoryRiskClassifier,
    NaiveInventoryRiskBaseline,
    compute_inventory_risk_score,
)
from src.risk_models.supplier_risk import (
    HistoricalSupplierBaseline,
    SupplierRiskClassifier,
    compute_supplier_risk_score,
)
from src.risk_models.exposure import (
    build_operational_exposure_layer,
)
from src.risk_models.explainability import (
    explain_operational_risk,
    generate_deterministic_evidence,
    generate_recommended_action,
)

__all__ = [
    "calculate_top_k_capture",
    "compute_brier_skill_score",
    "evaluate_risk_classification",
    "INVENTORY_FEATURE_WHITELIST",
    "SUPPLIER_FEATURE_WHITELIST",
    "assert_risk_feature_whitelist",
    "verify_future_perturbation_invariance",
    "InventoryRiskClassifier",
    "NaiveInventoryRiskBaseline",
    "compute_inventory_risk_score",
    "HistoricalSupplierBaseline",
    "SupplierRiskClassifier",
    "compute_supplier_risk_score",
    "build_operational_exposure_layer",
    "explain_operational_risk",
    "generate_deterministic_evidence",
    "generate_recommended_action",
]
