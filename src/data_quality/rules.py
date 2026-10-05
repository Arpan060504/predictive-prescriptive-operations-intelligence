"""
Data Quality Validation Rules for PPOI.
Defines deterministic validation rules across schema, domain constraints,
referential integrity, temporal continuity, and statistical anomaly detection.
Distinguishes severity levels: ERROR (critical), WARNING (suspicious), INFO (informational).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


class Severity(str, Enum):
    ERROR = "ERROR"       # Critical constraint failure requiring quarantine or repair
    WARNING = "WARNING"   # Plausible anomaly or data gap requiring investigation
    INFO = "INFO"         # Informational operational characteristic


@dataclass
class QualityIssue:
    rule_id: str
    rule_name: str
    table_name: str
    severity: Severity
    row_count: int
    sample_indices: List[Any]
    column_name: Optional[str]
    description: str
    action_suggested: str


def check_not_null(df: pd.DataFrame, table_name: str, key_columns: List[str]) -> List[QualityIssue]:
    """Verifies that primary and critical foreign keys contain no null values."""
    issues = []
    for col in key_columns:
        if col in df.columns:
            null_mask = df[col].isna() | (df[col].astype(str).str.strip().isin(["", "nan", "None", "NULL"]))
            n_null = int(null_mask.sum())
            if n_null > 0:
                issues.append(QualityIssue(
                    rule_id="DQ-001",
                    rule_name="NOT_NULL_KEYS",
                    table_name=table_name,
                    severity=Severity.ERROR,
                    row_count=n_null,
                    sample_indices=df.index[null_mask][:5].tolist(),
                    column_name=col,
                    description=f"Column '{col}' in '{table_name}' contains {n_null} null or blank values.",
                    action_suggested="Quarantine and remove records with missing mandatory identifiers."
                ))
    return issues


def check_valid_dates(df: pd.DataFrame, table_name: str, date_columns: List[str]) -> List[QualityIssue]:
    """Verifies that date columns conform to ISO YYYY-MM-DD format within valid calendar ranges."""
    issues = []
    for col in date_columns:
        if col in df.columns:
            # Attempt parsing
            parsed = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
            invalid_mask = parsed.isna() & df[col].notna()
            # Also check impossible dates like year < 2020 or year > 2030
            year_mask = parsed.notna() & ((parsed.dt.year < 2020) | (parsed.dt.year > 2030))
            total_invalid = invalid_mask | year_mask
            n_invalid = int(total_invalid.sum())
            if n_invalid > 0:
                issues.append(QualityIssue(
                    rule_id="DQ-002",
                    rule_name="VALID_DATE_FORMAT",
                    table_name=table_name,
                    severity=Severity.ERROR,
                    row_count=n_invalid,
                    sample_indices=df.index[total_invalid][:5].tolist(),
                    column_name=col,
                    description=f"Column '{col}' in '{table_name}' has {n_invalid} invalid or out-of-range dates.",
                    action_suggested="Quarantine unparseable dates or restore from primary calendar if deterministic."
                ))
    return issues


def check_duplicates(df: pd.DataFrame, table_name: str, subset_columns: Optional[List[str]] = None) -> List[QualityIssue]:
    """Verifies uniqueness across exact rows or primary key combinations."""
    issues = []
    dup_mask = df.duplicated(subset=subset_columns, keep="first")
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        issues.append(QualityIssue(
            rule_id="DQ-003",
            rule_name="NO_DUPLICATE_ROWS",
            table_name=table_name,
            severity=Severity.ERROR,
            row_count=n_dups,
            sample_indices=df.index[dup_mask][:5].tolist(),
            column_name=", ".join(subset_columns) if subset_columns else "ALL",
            description=f"Table '{table_name}' contains {n_dups} duplicate records.",
            action_suggested="Deduplicate records, retaining first authenticated entry."
        ))
    return issues


def check_non_negative(df: pd.DataFrame, table_name: str, numeric_columns: List[str]) -> List[QualityIssue]:
    """Verifies that non-negative physical quantities (demand, inventory, price) are >= 0."""
    issues = []
    for col in numeric_columns:
        if col in df.columns:
            # Filter non-null numerics
            s = pd.to_numeric(df[col], errors="coerce")
            neg_mask = s < 0
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                issues.append(QualityIssue(
                    rule_id="DQ-004",
                    rule_name="NON_NEGATIVE_NUMERICS",
                    table_name=table_name,
                    severity=Severity.ERROR,
                    row_count=n_neg,
                    sample_indices=df.index[neg_mask][:5].tolist(),
                    column_name=col,
                    description=f"Column '{col}' in '{table_name}' contains {n_neg} negative values.",
                    action_suggested="Repair by zero-clamping or absolute correction, logging the adjustment."
                ))
    return issues


def check_referential_integrity(
    child_df: pd.DataFrame,
    child_table: str,
    child_fk_col: str,
    parent_df: pd.DataFrame,
    parent_pk_col: str
) -> List[QualityIssue]:
    """Verifies that foreign key values in child tables strictly exist in parent dimension masters."""
    issues = []
    if child_fk_col in child_df.columns and parent_pk_col in parent_df.columns:
        valid_pks = set(parent_df[parent_pk_col].dropna().unique())
        orphan_mask = child_df[child_fk_col].notna() & (~child_df[child_fk_col].isin(valid_pks))
        n_orphans = int(orphan_mask.sum())
        if n_orphans > 0:
            issues.append(QualityIssue(
                rule_id="DQ-005",
                rule_name="REFERENTIAL_INTEGRITY",
                table_name=child_table,
                severity=Severity.ERROR,
                row_count=n_orphans,
                sample_indices=child_df.index[orphan_mask][:5].tolist(),
                column_name=child_fk_col,
                description=f"Foreign key '{child_fk_col}' in '{child_table}' contains {n_orphans} orphan records not found in parent master.",
                action_suggested="Quarantine orphan records that cannot be joined to master dimensions."
            ))
    return issues


def check_temporal_continuity(
    df: pd.DataFrame,
    table_name: str,
    group_cols: List[str],
    date_col: str = "date",
    expected_days: int = 731
) -> List[QualityIssue]:
    """Identifies sudden gaps in time-series sequences for SKU-warehouse or SKU-market combinations."""
    issues = []
    if date_col in df.columns and all(c in df.columns for c in group_cols):
        # Count dates per group
        counts = df.groupby(group_cols)[date_col].nunique()
        gap_groups = counts[counts < expected_days]
        if len(gap_groups) > 0:
            issues.append(QualityIssue(
                rule_id="DQ-006",
                rule_name="TEMPORAL_CONTINUITY",
                table_name=table_name,
                severity=Severity.WARNING,
                row_count=len(gap_groups),
                sample_indices=list(gap_groups.index[:5]),
                column_name=date_col,
                description=f"Detected {len(gap_groups)} time-series entities with missing calendar dates (< {expected_days} observations).",
                action_suggested="Re-index to continuous calendar, forward-filling or imputing missing daily observations."
            ))
    return issues


def check_statistical_outliers(
    df: pd.DataFrame,
    table_name: str,
    value_col: str,
    group_col: Optional[str] = None,
    z_threshold: float = 4.5
) -> List[QualityIssue]:
    """Detects statistical outliers exceeding z-score threshold within group distributions."""
    issues = []
    if value_col in df.columns:
        s = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        if group_col and group_col in df.columns:
            # Grouped z-score
            grouped = df.groupby(group_col)[value_col]
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0, 1.0)
            z_scores = ((s - mean) / std).abs()
        else:
            mean = s.mean()
            std = s.std() if s.std() > 0 else 1.0
            z_scores = ((s - mean) / std).abs()
            
        outlier_mask = z_scores > z_threshold
        n_outliers = int(outlier_mask.sum())
        if n_outliers > 0:
            issues.append(QualityIssue(
                rule_id="DQ-007",
                rule_name="STATISTICAL_OUTLIERS",
                table_name=table_name,
                severity=Severity.WARNING,
                row_count=n_outliers,
                sample_indices=df.index[outlier_mask][:5].tolist(),
                column_name=value_col,
                description=f"Found {n_outliers} statistical outliers (|z| > {z_threshold}) in '{value_col}'.",
                action_suggested="Flag for anomaly analysis; do not remove valid physical demand spikes."
            ))
    return issues
