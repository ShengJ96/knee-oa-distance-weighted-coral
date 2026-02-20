"""Evaluation metrics helpers for classification."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probs: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    metrics["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    if num_classes is None:
        num_classes = int(np.max(y_true)) + 1
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    totals = cm.sum()
    if totals > 0:
        for idx in range(num_classes):
            tp = float(cm[idx, idx])
            fn = float(cm[idx, :].sum() - tp)
            fp = float(cm[:, idx].sum() - tp)
            tn = float(totals - tp - fn - fp)
            metrics[f"tpr_class_{idx}"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[f"tnr_class_{idx}"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if probs is not None and probs.shape[0] == y_true.shape[0]:
        for threshold in (2, 3):
            if threshold >= num_classes:
                continue
            binary_true = (y_true >= threshold).astype(int)
            if np.unique(binary_true).size < 2:
                continue
            pos_prob = probs[:, threshold:].sum(axis=1)
            try:
                metrics[f"auc_ge{threshold}"] = float(
                    roc_auc_score(binary_true, pos_prob)
                )
            except ValueError:
                continue

    return metrics


def confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, str]:
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)
    return cm, report
