"""
PPOI SQL Analytics & Analytical Views Package.
Exposes view builders, analytical query abstractions, KPI metrics registry,
feature engineering bases, and reconciliation validation audits.
"""

from src.analytics.metrics import KPI_REGISTRY, KPIDefinition
from src.analytics.views import SQL_VIEW_DEFINITIONS, build_materialized_analytical_tables
from src.analytics.feature_base import build_demand_feature_dataset, verify_temporal_leakage_invariance
from src.analytics.queries import (
    query_executive_kpis,
    query_category_performance,
    query_monthly_cost_breakdown,
    query_supplier_scorecard,
    query_warehouse_capacity_and_utilization,
    query_demand_volatility_and_concentration,
    query_stockout_duration_distribution,
)
from src.analytics.validation import (
    verify_demand_reconciliation,
    verify_inventory_reconciliation,
    verify_cost_summary_reconciliation,
    verify_anti_cartesian_protection,
    verify_cross_phase_demand_reconciliation,
    verify_cross_phase_cost_reconciliation,
    verify_materialized_grains,
    run_full_analytical_audit,
)

__all__ = [
    "KPI_REGISTRY",
    "KPIDefinition",
    "SQL_VIEW_DEFINITIONS",
    "build_materialized_analytical_tables",
    "build_demand_feature_dataset",
    "verify_temporal_leakage_invariance",
    "query_executive_kpis",
    "query_category_performance",
    "query_monthly_cost_breakdown",
    "query_supplier_scorecard",
    "query_warehouse_capacity_and_utilization",
    "query_demand_volatility_and_concentration",
    "query_stockout_duration_distribution",
    "verify_demand_reconciliation",
    "verify_inventory_reconciliation",
    "verify_cost_summary_reconciliation",
    "verify_anti_cartesian_protection",
    "verify_cross_phase_demand_reconciliation",
    "verify_cross_phase_cost_reconciliation",
    "verify_materialized_grains",
    "run_full_analytical_audit",
]
