"""
Test Suite for Phase 3: Data Quality Engine & Remediation Pipeline.
Verifies quality rule validators, issue injection tracking, automated cleaning behavior,
before/after metrics reduction to zero critical errors, referential integrity, and determinism.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.utils.config import get_project_root
from src.data_quality.rules import (
    Severity,
    check_not_null,
    check_valid_dates,
    check_duplicates,
    check_non_negative,
    check_referential_integrity,
    check_temporal_continuity,
    check_statistical_outliers,
)
from src.data_quality.auditor import run_quality_audit, audit_table_collection


@pytest.fixture(scope="module")
def project_paths():
    root = get_project_root()
    return {
        "raw": root / "data" / "raw",
        "raw_issues": root / "data" / "raw_with_quality_issues",
        "processed": root / "data" / "processed",
        "json_report": root / "data_quality_report.json",
        "md_report": root / "reports" / "data_quality_report.md",
        "injection_log": root / "data" / "raw_with_quality_issues" / "injected_issues_log.csv",
        "audit_trail": root / "data" / "processed" / "audit_trail_log.csv"
    }


def test_quality_files_and_reports_exist(project_paths):
    """Verify that all quality issue files, cleaned tables, and audit logs exist."""
    assert project_paths["raw_issues"].exists(), "Directory raw_with_quality_issues missing"
    assert project_paths["processed"].exists(), "Directory data/processed missing"
    assert project_paths["json_report"].exists(), "data_quality_report.json missing"
    assert project_paths["md_report"].exists(), "reports/data_quality_report.md missing"
    assert project_paths["injection_log"].exists(), "injected_issues_log.csv missing"
    assert project_paths["audit_trail"].exists(), "audit_trail_log.csv missing"


def test_individual_validation_rules():
    """Verify individual validator rule logic against synthetic test cases."""
    # 1. Not null rule
    df_null = pd.DataFrame({"id": ["A", None, "C", ""]})
    issues = check_not_null(df_null, "test_table", ["id"])
    assert len(issues) == 1
    assert issues[0].row_count == 2
    assert issues[0].severity == Severity.ERROR

    # 2. Date format rule
    df_date = pd.DataFrame({"date": ["2024-01-01", "2024-02-30", "INVALID", "2099-01-01"]})
    issues = check_valid_dates(df_date, "test_table", ["date"])
    assert len(issues) == 1
    assert issues[0].row_count == 3 # 2024-02-30 invalid, INVALID invalid, 2099 out of range

    # 3. Duplicate rows rule
    df_dup = pd.DataFrame({"k": [1, 2, 2, 3], "v": ["a", "b", "b", "c"]})
    issues = check_duplicates(df_dup, "test_table")
    assert len(issues) == 1
    assert issues[0].row_count == 1

    # 4. Non-negative numerics rule
    df_neg = pd.DataFrame({"val": [10, -5, 0, -12]})
    issues = check_non_negative(df_neg, "test_table", ["val"])
    assert len(issues) == 1
    assert issues[0].row_count == 2

    # 5. Referential integrity rule
    df_child = pd.DataFrame({"pid": ["SKU-01", "SKU-99", "SKU-02"]})
    df_parent = pd.DataFrame({"pid": ["SKU-01", "SKU-02", "SKU-03"]})
    issues = check_referential_integrity(df_child, "child", "pid", df_parent, "pid")
    assert len(issues) == 1
    assert issues[0].row_count == 1
    assert issues[0].sample_indices == [1]


def test_injection_log_traceability(project_paths):
    """Verify that every injected flaw has full traceability in injected_issues_log.csv."""
    df_inj = pd.read_csv(project_paths["injection_log"])
    assert len(df_inj) > 200, "Must have tracked injected flaws"
    required_cols = [
        "issue_id", "table_name", "row_index", "column_name",
        "original_value", "corrupted_value", "issue_category", "severity", "description"
    ]
    for col in required_cols:
        assert col in df_inj.columns, f"Column '{col}' missing from injected_issues_log.csv"

    # Verify categories present
    cats = set(df_inj["issue_category"].unique())
    assert "NULL_KEY" in cats
    assert "INVALID_DATE" in cats
    assert "DUPLICATE_RECORD" in cats
    assert "NEGATIVE_NUMERIC" in cats
    assert "IMPOSSIBLE_INVENTORY" in cats
    assert "REFERENTIAL_VIOLATION" in cats


def test_pre_cleaning_detection(project_paths):
    """Verify that auditing raw_with_quality_issues detects critical errors."""
    issues, _ = audit_table_collection(project_paths["raw_issues"])
    error_count = sum(i.row_count for i in issues if i.severity == Severity.ERROR)
    warning_count = sum(i.row_count for i in issues if i.severity == Severity.WARNING)

    assert error_count > 0, "Pre-cleaning audit must detect critical errors"
    assert warning_count > 0, "Pre-cleaning audit must detect warnings"


def test_post_cleaning_zero_critical_errors(project_paths):
    """Verify that processed analytical tables have exactly zero critical errors."""
    issues, summaries = audit_table_collection(project_paths["processed"])
    error_count = sum(i.row_count for i in issues if i.severity == Severity.ERROR)

    assert error_count == 0, f"Processed dataset must have 0 critical errors, found {error_count}"

    # Verify all tables in processed have 0% missing and 0% duplicates
    for tbl_name, summary in summaries.items():
        assert summary["missing_pct"] == 0.0, f"Table '{tbl_name}' still has missing values"
        assert summary["duplicate_pct"] == 0.0, f"Table '{tbl_name}' still has duplicates"


def test_post_cleaning_referential_integrity(project_paths):
    """Verify 100% referential integrity in cleaned processed tables."""
    df_products = pd.read_csv(project_paths["processed"] / "products.csv")
    df_suppliers = pd.read_csv(project_paths["processed"] / "suppliers.csv")
    df_warehouses = pd.read_csv(project_paths["processed"] / "warehouses.csv")
    df_demand = pd.read_csv(project_paths["processed"] / "demand.csv")
    df_pos = pd.read_csv(project_paths["processed"] / "purchase_orders.csv")
    df_inventory = pd.read_csv(project_paths["processed"] / "inventory.csv")

    valid_pids = set(df_products["product_id"])
    valid_sids = set(df_suppliers["supplier_id"])
    valid_wids = set(df_warehouses["warehouse_id"])

    assert set(df_demand["product_id"]).issubset(valid_pids)
    assert set(df_pos["supplier_id"]).issubset(valid_sids)
    assert set(df_pos["product_id"]).issubset(valid_pids)
    assert set(df_inventory["product_id"]).issubset(valid_pids)
    assert set(df_inventory["warehouse_id"]).issubset(valid_wids)


def test_audit_trail_non_silent_cleaning(project_paths):
    """Verify that every repair and quarantine action has a recorded business rationale."""
    df_audit = pd.read_csv(project_paths["audit_trail"])
    assert len(df_audit) > 0, "Audit trail log cannot be empty"

    assert "action" in df_audit.columns
    assert "records_affected" in df_audit.columns
    assert "reason" in df_audit.columns
    assert "disposition" in df_audit.columns

    # Verify both REMOVED and REPAIRED actions are present
    dispositions = set(df_audit["disposition"].unique())
    assert "REMOVED" in dispositions
    assert "REPAIRED" in dispositions


def test_clean_operational_reconciliation(project_paths):
    """
    Ensure the processed inventory table reconciles strictly:
    Beginning Inventory + PO Received - Demand Fulfilled - Ending Inventory == 0
    """
    df_inv = pd.read_csv(project_paths["processed"] / "inventory.csv")
    diff = (
        df_inv["beginning_inventory"]
        + df_inv["po_received"]
        - df_inv["demand_fulfilled"]
        - df_inv["ending_inventory"]
    ).abs()
    assert (diff == 0).all(), "Processed inventory table failed balance equation"
    assert (df_inv["ending_inventory"] >= 0).all(), "Negative ending inventory in processed"


def test_injection_count_reconciliation(project_paths):
    """
    Verify exact reconciliation of the 511 injected issue instances in injected_issues_log.csv:
    60 null demand + 35 null PO + 45 invalid dates + 120 duplicate demand + 50 duplicate PO +
    25 conflicting PO keys + 30 negative demand + 15 negative PO + 40 impossible inventory +
    50 orphan demand + 30 orphan PO + 1 temporal gap + 10 statistical outliers = 511 instances.
    """
    df_inj = pd.read_csv(project_paths["injection_log"])
    assert len(df_inj) == 511, f"Expected 511 logged injection instances, found {len(df_inj)}"

    cat_counts = df_inj["issue_category"].value_counts().to_dict()
    assert cat_counts["NULL_KEY"] == 95 # 60 demand + 35 PO
    assert cat_counts["INVALID_DATE"] == 45
    assert cat_counts["DUPLICATE_RECORD"] == 170 # 120 demand + 50 PO
    assert cat_counts["DUPLICATE_TRANSACTION_KEY"] == 25
    assert cat_counts["NEGATIVE_NUMERIC"] == 45 # 30 demand + 15 PO
    assert cat_counts["IMPOSSIBLE_INVENTORY"] == 40
    assert cat_counts["REFERENTIAL_VIOLATION"] == 80 # 50 demand + 30 PO
    assert cat_counts["TEMPORAL_GAP"] == 1
    assert cat_counts["STATISTICAL_OUTLIER"] == 10
    assert sum(cat_counts.values()) == 511


def test_warning_before_after_reconciliation(project_paths):
    """
    Reconcile the warning count transition from 697 pre-remediation to 701 post-remediation:
    - DQ-006 (TEMPORAL_CONTINUITY): increases from 36 to 41 (+5) because removing invalid dates
      reduced 5 SKU-warehouse series from 731 to 730 valid calendar observations.
    - DQ-007 (STATISTICAL_OUTLIERS): decreases from 661 to 660 (-1) because 1 outlier row
      was on an invalid/orphan key row that was quarantined.
    - Net change: +5 - 1 = +4 (697 -> 701).
    """
    with open(project_paths["json_report"], "r", encoding="utf-8") as f:
        rep = json.load(f)

    pre_warnings = {i["rule_id"]: i["row_count"] for i in rep["pre_issues_detected"] if i["severity"] == "WARNING"}
    post_warnings = {i["rule_id"]: i["row_count"] for i in rep["post_issues_detected"] if i["severity"] == "WARNING"}

    assert pre_warnings["DQ-006"] == 36
    assert pre_warnings["DQ-007"] == 661
    assert sum(pre_warnings.values()) == 697

    assert post_warnings["DQ-006"] == 41
    assert post_warnings["DQ-007"] == 660
    assert sum(post_warnings.values()) == 701


def test_outlier_injected_vs_natural_distinction(project_paths):
    """
    Verify the explicit breakdown of the 577 retained and flagged statistical demand outliers:
    - Exactly 10 were deliberately injected artificial spikes (demand_requested = 3200 units).
    - Exactly 567 are naturally occurring operational peak surges (holiday rush, promotional lifts,
      industrial maintenance waves) with z > 5.0 in the synthetic environment.
    """
    df_proc_demand = pd.read_csv(project_paths["processed"] / "demand.csv")
    outliers = df_proc_demand[df_proc_demand["outlier_flag"] == 1]
    assert len(outliers) == 577, f"Expected 577 flagged outliers, found {len(outliers)}"

    injected_spikes = outliers[outliers["demand_requested"] >= 3000]
    natural_surges = outliers[outliers["demand_requested"] < 3000]

    assert len(injected_spikes) == 10, f"Expected exactly 10 injected artificial spikes, found {len(injected_spikes)}"
    assert len(natural_surges) == 567, f"Expected exactly 567 naturally occurring peak surges, found {len(natural_surges)}"


def test_row_count_reconciliation(project_paths):
    """
    Verify exact accounting from baseline clean data to corrupted data to cleaned processed data.
    """
    df_raw_dem = pd.read_csv(project_paths["raw"] / "demand.csv")
    df_iss_dem = pd.read_csv(project_paths["raw_issues"] / "demand.csv")
    df_proc_dem = pd.read_csv(project_paths["processed"] / "demand.csv")

    df_raw_pos = pd.read_csv(project_paths["raw"] / "purchase_orders.csv")
    df_iss_pos = pd.read_csv(project_paths["raw_issues"] / "purchase_orders.csv")
    df_proc_pos = pd.read_csv(project_paths["processed"] / "purchase_orders.csv")

    df_raw_trans = pd.read_csv(project_paths["raw"] / "transport.csv")
    df_proc_trans = pd.read_csv(project_paths["processed"] / "transport.csv")

    df_raw_inv = pd.read_csv(project_paths["raw"] / "inventory.csv")
    df_proc_inv = pd.read_csv(project_paths["processed"] / "inventory.csv")

    # Demand: 263160 + 120 (dups) - 28 (gap) = 263252 -> 262978 (-120 dups, -60 nulls, -50 orphans, -44 invalid dates)
    assert len(df_raw_dem) == 263160
    assert len(df_iss_dem) == 263252
    assert len(df_proc_dem) == 262978
    assert len(df_iss_dem) - len(df_proc_dem) == 274 # 120 + 60 + 50 + 44

    # Purchase Orders: 11730 + 50 (dups) + 25 (conflicts) = 11805 -> 11665 (-50 dups, -25 conflicts, -35 nulls, -30 orphans)
    assert len(df_raw_pos) == 11730
    assert len(df_iss_pos) == 11805
    assert len(df_proc_pos) == 11665
    assert len(df_iss_pos) - len(df_proc_pos) == 140 # 50 + 25 + 35 + 30

    # Transport: 11730 baseline - 65 cascading PO quarantines = 11665
    assert len(df_raw_trans) == 11730
    assert len(df_proc_trans) == 11665
    assert len(df_raw_trans) - len(df_proc_trans) == 65

    # Inventory: 175440 baseline -> 175440 processed (0 removed, 40 repaired)
    assert len(df_raw_inv) == 175440
    assert len(df_proc_inv) == 175440


def test_cascading_po_transport_quarantine(project_paths):
    """Verify that every quarantined PO caused its matching transport record to be quarantined."""
    df_proc_pos = pd.read_csv(project_paths["processed"] / "purchase_orders.csv")
    df_proc_trans = pd.read_csv(project_paths["processed"] / "transport.csv")

    valid_pos = set(df_proc_pos["po_id"])
    assert set(df_proc_trans["po_id"]).issubset(valid_pos), "Orphan transport shipments detected"
    assert len(df_proc_pos) == len(df_proc_trans) == 11665
