"""
Logging utility for PPOI.
Standardizes console formatting across all ETL, training, and simulation pipelines.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "PPOI", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
