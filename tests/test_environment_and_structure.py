"""
Test Suite for Phase 1: Repository Architecture, Environment, and Configuration.
Verifies project directories, config integrity, core library imports, and utility functions.
"""

import sys
from pathlib import Path
import pytest
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root, load_config, get_resolved_path
from src.utils.logger import get_logger


def test_project_root_resolution():
    """Verify that get_project_root returns the exact directory containing run.py."""
    root = get_project_root()
    assert root.exists(), "Project root does not exist"
    assert (root / "run.py").exists(), "run.py not found in project root"
    assert (root / "config" / "config.yaml").exists(), "config.yaml not found"


def test_directory_structure():
    """Verify all mandatory subdirectories exist."""
    root = get_project_root()
    required_dirs = [
        "config",
        "data/raw",
        "data/processed",
        "data/demo",
        "database",
        "src/data_generation",
        "src/data_quality",
        "src/database",
        "src/feature_engineering",
        "src/forecasting",
        "src/risk_models",
        "src/simulation",
        "src/optimization",
        "src/decision_engine",
        "src/utils",
        "models",
        "dashboard/pages",
        "dashboard/components",
        "tests",
        "reports"
    ]
    for d in required_dirs:
        path = root / d
        assert path.exists() and path.is_dir(), f"Missing required directory: {d}"


def test_config_content():
    """Verify configuration YAML schema and default values."""
    config = load_config()
    assert isinstance(config, dict), "Config must be a dictionary"
    
    # Check top-level sections
    assert "project" in config
    assert "paths" in config
    assert "data_generation" in config
    assert "forecasting" in config
    assert "risk_models" in config
    assert "optimization" in config
    assert "simulation" in config

    # Check synthetic disclosure
    disclosure = config["project"].get("synthetic_disclosure", "")
    assert "synthetic" in disclosure.lower(), "Synthetic data disclosure missing in config"

    # Check forecasting hierarchy
    expected_hierarchy = ["seasonal_naive", "moving_average", "ridge", "xgboost"]
    assert config["forecasting"]["hierarchy"] == expected_hierarchy, (
        f"Forecasting hierarchy must be {expected_hierarchy}"
    )

    # Check optimization solver
    assert config["optimization"]["solver"] == "highs"


def test_core_dependencies_import():
    """Verify that all core numerical, ML, and visualization packages import cleanly."""
    import pandas as pd
    import numpy as np
    import scipy
    from scipy.optimize import linprog, milp
    import sklearn
    import xgboost as xgb
    import plotly
    import streamlit as st
    import sqlite3

    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert scipy.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgb.__version__ is not None
    assert plotly.__version__ is not None


def test_logger():
    """Verify that centralized logger initializes and functions properly."""
    logger = get_logger("TestLogger")
    assert logger is not None
    assert logger.name == "TestLogger"
    logger.info("Test logger message emitted successfully.")
