"""
Tests for Centralized Database Resolver & Demo Database Deployment.
Validates resolution precedence, fallback mechanism, environment overrides,
connection integrity, and demo database compliance (< 100 MB, referential integrity).
"""

import os
from pathlib import Path
import sqlite3
import pytest

from src.database.connection import (
    PROJECT_ROOT,
    FULL_DB_PATH,
    DEMO_DB_PATH,
    get_database_path,
    is_demo_database,
    get_database_mode_label,
    get_connection,
    get_database_info,
)


def test_database_path_resolution_default():
    """Verify that get_database_path resolves to an existing database file."""
    db_path = get_database_path()
    assert db_path.exists(), f"Resolved database does not exist: {db_path}"
    # In standard local setup, operations.db takes precedence
    if FULL_DB_PATH.exists():
        assert db_path == FULL_DB_PATH
        assert is_demo_database(db_path) is False
        assert get_database_mode_label(db_path) == "FULL DATABASE"


def test_database_path_resolution_fallback_to_demo(monkeypatch):
    """Verify fallback to demo_operations.db when operations.db is missing."""
    # Simulate environment where operations.db does not exist
    monkeypatch.setattr(Path, "exists", lambda self: self == DEMO_DB_PATH)
    resolved = get_database_path()
    assert resolved == DEMO_DB_PATH
    assert is_demo_database(resolved) is True
    assert get_database_mode_label(resolved) == "DEPLOYMENT DEMO DATABASE"


def test_database_path_resolution_force_demo(monkeypatch):
    """Verify forced demo mode via PPOI_FORCE_DEMO_DB environment variable."""
    assert DEMO_DB_PATH.exists(), "demo_operations.db must exist for forced demo test"
    monkeypatch.setenv("PPOI_FORCE_DEMO_DB", "1")
    resolved = get_database_path()
    assert resolved == DEMO_DB_PATH
    assert is_demo_database(resolved) is True
    assert get_database_mode_label(resolved) == "DEPLOYMENT DEMO DATABASE"


def test_database_path_resolution_neither_exists(monkeypatch):
    """Verify FileNotFoundError is raised with helpful message when neither DB exists."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError) as exc_info:
        get_database_path()
    assert "Operational database not found" in str(exc_info.value)
    assert "demo_operations.db" in str(exc_info.value)


def test_get_connection_foreign_keys_and_query():
    """Verify get_connection returns functional SQLite connection with foreign keys enabled."""
    conn = get_connection(readonly=False)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys;")
        fk_status = cur.fetchone()[0]
        assert fk_status == 1, "Foreign keys pragma must be ON (1)"

        cur.execute("SELECT COUNT(*) FROM dim_product;")
        product_count = cur.fetchone()[0]
        assert product_count == 60, f"Expected 60 products, got {product_count}"
    finally:
        conn.close()


def test_get_database_info_structure():
    """Verify get_database_info returns complete telemetry dictionary."""
    info = get_database_info()
    assert "path" in info
    assert "filename" in info
    assert "is_demo" in info
    assert "mode_label" in info
    assert "size_mb" in info
    assert "tables_count" in info
    assert info["size_mb"] > 0
    assert info["tables_count"] >= 20


def test_demo_database_file_and_size_limits():
    """Verify demo_operations.db exists, is <100MB (and <25MB), and has valid integrity."""
    assert DEMO_DB_PATH.exists(), "database/demo_operations.db does not exist"
    size_bytes = DEMO_DB_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Strict GitHub limit
    assert size_mb < 100.0, f"demo_operations.db ({size_mb:.2f} MB) exceeds GitHub 100MB limit"
    # Target deployment size
    assert size_mb < 25.0, f"demo_operations.db ({size_mb:.2f} MB) exceeds target 25MB limit"


def test_demo_database_schema_and_fk_integrity():
    """Verify demo_operations.db schema, tables, and 0 foreign key violations."""
    conn = sqlite3.connect(str(DEMO_DB_PATH))
    try:
        cur = conn.cursor()
        # 1. Referential integrity
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("PRAGMA foreign_key_check;")
        fk_violations = cur.fetchall()
        assert len(fk_violations) == 0, f"Demo DB has foreign key violations: {fk_violations}"

        # 2. Dimensions
        cur.execute("SELECT COUNT(*) FROM dim_product;")
        assert cur.fetchone()[0] == 60
        cur.execute("SELECT COUNT(*) FROM dim_supplier;")
        assert cur.fetchone()[0] == 8
        cur.execute("SELECT COUNT(*) FROM dim_warehouse;")
        assert cur.fetchone()[0] == 4

        # 3. Prescriptive optimization tables
        cur.execute("SELECT COUNT(*) FROM optimization_decisions;")
        assert cur.fetchone()[0] > 0
        cur.execute("SELECT COUNT(*) FROM scenario_definitions;")
        assert cur.fetchone()[0] == 6

        # 4. Analytics exposure and risk tables
        cur.execute("SELECT COUNT(*) FROM analytics_operational_exposure;")
        assert cur.fetchone()[0] > 0
        cur.execute("SELECT COUNT(*) FROM analytics_supplier_risk;")
        assert cur.fetchone()[0] > 0
    finally:
        conn.close()
