"""
Configuration and Path Management for PPOI.
Resolves paths relative to the project root regardless of execution context.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Determine project root (2 levels above src/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Returns the absolute Path to the project root directory."""
    return PROJECT_ROOT


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration file and resolves all relative directory paths to absolute Paths.
    """
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_resolved_path(relative_path: str) -> Path:
    """Resolves any relative path against the project root."""
    return PROJECT_ROOT / relative_path
