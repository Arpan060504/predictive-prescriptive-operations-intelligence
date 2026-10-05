"""
Risk Classification & Calibration Evaluation Metrics for PPOI (Phase 8).
Provides mathematically robust metrics for assessing probability calibration,
discrimination, and operational top-decile capture under class imbalance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


def calculate_top_k_capture(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    k_percent: float = 0.10,
) -> float:
    """
    Computes the proportion of total true positive events captured in the top k%
    highest predicted risk probabilities.
    
    Formula:
        Capture Rate = (True Positives in Top K%) / (Total True Positives)
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    
    total_positives = np.sum(y_t)
    if total_positives == 0:
        return 0.0
    
    n_samples = len(y_t)
    top_n = max(1, int(np.ceil(n_samples * k_percent)))
    
    # Sort descending by predicted probability
    top_indices = np.argsort(y_p)[::-1][:top_n]
    positives_captured = np.sum(y_t[top_indices])
    
    return float(positives_captured / total_positives)


def compute_brier_skill_score(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> float:
    """
    Computes Brier Skill Score (BSS) relative to the empirical event rate baseline:
        BSS = 1 - (Brier_model / Brier_climatology)
    where Brier_climatology is the Brier score of predicting the base rate for all instances.
    """
    y_t = np.asarray(y_true).astype(float)
    y_p = np.asarray(y_prob).astype(float)
    
    base_rate = np.mean(y_t)
    brier_ref = np.mean((base_rate - y_t) ** 2)
    brier_mod = brier_score_loss(y_t, y_p)
    
    if brier_ref <= 1e-12:
        return 0.0
    return float(1.0 - (brier_mod / brier_ref))


def evaluate_risk_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of risk classification and calibration metrics.
    
    Handles class imbalance by prioritizing PR-AUC, Brier score, and top-decile capture
    over standard accuracy.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    
    # Probability bounds clipping
    y_p_clipped = np.clip(y_p, 1e-7, 1.0 - 1e-7)
    y_pred = (y_p >= threshold).astype(int)
    
    n_samples = len(y_t)
    event_count = int(np.sum(y_t))
    event_rate = float(event_count / max(n_samples, 1))
    
    # Discrimination metrics
    try:
        roc_auc = float(roc_auc_score(y_t, y_p))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_t, y_p))
    except Exception:
        pr_auc = event_rate
        
    # Calibration metrics
    brier = float(brier_score_loss(y_t, y_p_clipped))
    brier_skill = compute_brier_skill_score(y_t, y_p_clipped)
    
    try:
        ce_loss = float(log_loss(y_t, y_p_clipped))
    except Exception:
        ce_loss = float("nan")
        
    # Threshold-dependent metrics
    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_t, y_pred)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else (int(cm[0, 0]) if y_t[0] == 0 else 0)
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else (int(cm[0, 0]) if y_t[0] == 1 else 0)
    
    # Operational priority capture rates
    top_5pct_capture = calculate_top_k_capture(y_t, y_p, k_percent=0.05)
    top_10pct_capture = calculate_top_k_capture(y_t, y_p, k_percent=0.10)
    top_20pct_capture = calculate_top_k_capture(y_t, y_p, k_percent=0.20)
    
    # Calibration reliability curve (5 bins)
    try:
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=5, strategy="uniform")
        calib_curve = {
            "prob_true": [float(v) for v in prob_true],
            "prob_pred": [float(v) for v in prob_pred],
        }
    except Exception:
        calib_curve = {"prob_true": [], "prob_pred": []}
        
    metrics = {
        f"{prefix}roc_auc": round(roc_auc, 4),
        f"{prefix}pr_auc": round(pr_auc, 4),
        f"{prefix}brier_score": round(brier, 4),
        f"{prefix}brier_skill_score": round(brier_skill, 4),
        f"{prefix}log_loss": round(ce_loss, 4),
        f"{prefix}precision": round(precision, 4),
        f"{prefix}recall": round(recall, 4),
        f"{prefix}f1": round(f1, 4),
        f"{prefix}top_5pct_capture": round(top_5pct_capture, 4),
        f"{prefix}top_10pct_capture": round(top_10pct_capture, 4),
        f"{prefix}top_20pct_capture": round(top_20pct_capture, 4),
        f"{prefix}event_rate": round(event_rate, 4),
        f"{prefix}n_samples": n_samples,
        f"{prefix}event_count": event_count,
        f"{prefix}confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        f"{prefix}calibration_curve": calib_curve,
    }
    return metrics
