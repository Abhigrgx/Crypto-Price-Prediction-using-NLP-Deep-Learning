"""
Model evaluation: regression and classification metrics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from loguru import logger


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Return RMSE, MAE, MAPE, R² for regression predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    metrics = {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}% | R²={r2:.4f}"
    )
    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, float]:
    """Return accuracy, F1, AUC for binary classification."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}Accuracy={acc:.4f} | F1={f1:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return {"accuracy": acc, "f1": f1}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of predictions where the direction (up/down) is correct.
    Requires consecutive-step arrays.
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    da = float(np.mean(true_dir == pred_dir))
    logger.info(f"Directional Accuracy: {da:.4f}")
    return da
