"""
Centralized Database Connection & Resolution Engine for PPOI.
Handles transparent resolution and connection management between the local certified
operational database (database/operations.db) and the lightweight deployment demo database
(database/demo_operations.db) for Streamlit Community Cloud and CI/CD environments.
"""

import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional

# Resolve project root (two levels above src/database)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FULL_DB_PATH = PROJECT_ROOT / "database" / "operations.db"
DEMO_DB_PATH = PROJECT_ROOT / "database" / "demo_operations.db"


def get_database_path() -> Path:
    """
    Resolves the active operational database path with fallback logic.
    
    Resolution Order:
    1. Environment variable override (PPOI_DATABASE_PATH).
    2. Forced demo mode via environment variable (PPOI_FORCE_DEMO_DB=1) if demo DB exists.
    3. Certified Full Database: database/operations.db (primary for local analytics).
    4. Lightweight Demo Database: database/demo_operations.db (primary for Streamlit Cloud deployment).
    5. Raises FileNotFoundError with actionable remediation instructions.
    """
    env_override = os.environ.get("PPOI_DATABASE_PATH")
    if env_override:
        custom_path = Path(env_override)
        if not custom_path.is_absolute():
            custom_path = PROJECT_ROOT / custom_path
        if custom_path.exists():
            return custom_path
        raise FileNotFoundError(
            f"Configured PPOI_DATABASE_PATH does not exist: {custom_path}"
        )

    force_demo = os.environ.get("PPOI_FORCE_DEMO_DB", "").lower() in ("1", "true", "yes")
    if force_demo and DEMO_DB_PATH.exists():
        return DEMO_DB_PATH

    # Standard Priority 1: Full operational database
    if FULL_DB_PATH.exists():
        return FULL_DB_PATH

    # Standard Priority 2: Demo operational database
    if DEMO_DB_PATH.exists():
        return DEMO_DB_PATH

    # Failure: neither exists
    raise FileNotFoundError(
        "Operational database not found. Neither certified full database "
        f"('{FULL_DB_PATH}') nor deployment demo database ('{DEMO_DB_PATH}') exists.\n"
        "Remediation:\n"
        "  - In local environments: run 'python run.py build-database' to compile operations.db\n"
        "  - In deployment environments: run 'python scripts/create_demo_database.py' "
        "to generate demo_operations.db"
    )


def is_demo_database(db_path: Optional[Path] = None) -> bool:
    """
    Determines whether the specified or resolved database is the deployment demo database.
    """
    if db_path is None:
        try:
            db_path = get_database_path()
        except FileNotFoundError:
            return False
    return db_path.name == "demo_operations.db"


def get_database_mode_label(db_path: Optional[Path] = None) -> str:
    """
    Returns human-readable deployment mode label for the active database.
    """
    return "DEPLOYMENT DEMO DATABASE" if is_demo_database(db_path) else "FULL DATABASE"


def get_connection(
    db_path: Optional[Path] = None,
    readonly: bool = False,
    timeout: float = 10.0,
) -> sqlite3.Connection:
    """
    Establishes an optimized SQLite connection to the active operational database.
    
    Args:
        db_path: Optional explicit database path. If None, resolves via get_database_path().
        readonly: If True, opens database in read-only mode via SQLite URI.
        timeout: SQLite lock timeout in seconds.
        
    Returns:
        Configured sqlite3.Connection with foreign keys enabled.
    """
    resolved_path = db_path if db_path is not None else get_database_path()
    resolved_path = resolved_path.resolve()

    if readonly:
        uri_path = f"file:{resolved_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(str(resolved_path), timeout=timeout)

    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_database_info() -> Dict[str, Any]:
    """
    Returns telemetry metadata regarding the active database connection.
    """
    path = get_database_path()
    is_demo = is_demo_database(path)
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    
    conn = get_connection(path, readonly=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        tables_count = cur.fetchone()[0]
    finally:
        conn.close()

    return {
        "path": str(path),
        "filename": path.name,
        "is_demo": is_demo,
        "mode_label": get_database_mode_label(path),
        "size_mb": size_mb,
        "tables_count": tables_count,
    }
