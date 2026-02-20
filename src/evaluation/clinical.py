"""Clinical utility metrics (calibration, DCA, selective prediction helpers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TemperatureScalingResult:
    temperature: float
    logits: torch.Tensor
    probs: torch.Tensor


def temperature_scale_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
    init_temp: float = 1.0,
) -> TemperatureScalingResult:
    """Fit a single scalar temperature using NLL minimization."""

    device = logits.device
    log_temp = torch.log(torch.tensor([init_temp], device=device))
    log_temp.requires_grad_(True)
    optimizer = torch.optim.LBFGS([log_temp], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temp).detach().clamp(min=1e-3, max=10.0).item())
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)
    return TemperatureScalingResult(temperature=temperature, logits=scaled_logits, probs=probs)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> Dict:
    """Compute Expected Calibration Error and per-bin stats."""

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True) - 1
    ece = 0.0
    bin_stats: List[Dict] = []
    total = len(labels)

    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        frac = float(np.mean(mask))
        bin_conf = float(confidences[mask].mean())
        bin_acc = float((predictions[mask] == labels[mask]).mean())
        ece += abs(bin_acc - bin_conf) * frac
        bin_stats.append(
            {
                "bin": idx,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "weight": frac,
            }
        )

    return {"ece": float(ece), "bins": bin_stats}


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> Dict[str, List[float]]:
    """Return calibration curve arrays (confidence vs accuracy)."""

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: List[float] = []
    accuracies: List[float] = []

    for start, end in zip(bins[:-1], bins[1:]):
        mask = (confidences >= start) & (confidences < end)
        if not np.any(mask):
            continue
        bin_centers.append(float((start + end) / 2.0))
        accuracies.append(float((predictions[mask] == labels[mask]).mean()))

    return {"confidence": bin_centers, "accuracy": accuracies}


def decision_curve_analysis(
    probabilities: np.ndarray,
    binary_labels: np.ndarray,
    thresholds: Iterable[float],
) -> Dict[str, List[Dict[str, float]]]:
    """Compute net benefit for prediction, treat-all, treat-none strategies."""

    thresholds = [float(t) for t in thresholds if 0.0 < t < 1.0]
    if not thresholds:
        raise ValueError("Decision Curve Analysis requires thresholds in (0, 1).")

    n = float(len(binary_labels))
    prevalence = float(np.mean(binary_labels))
    nb_model: List[Dict[str, float]] = []
    nb_treat_all: List[Dict[str, float]] = []
    nb_treat_none: List[Dict[str, float]] = []

    for threshold in thresholds:
        preds = probabilities >= threshold
        tp = float(np.sum((preds == 1) & (binary_labels == 1)))
        fp = float(np.sum((preds == 1) & (binary_labels == 0)))
        net = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
        treat_all = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
        nb_model.append({"threshold": threshold, "net_benefit": net})
        nb_treat_all.append({"threshold": threshold, "net_benefit": treat_all})
        nb_treat_none.append({"threshold": threshold, "net_benefit": 0.0})

    return {
        "model": nb_model,
        "treat_all": nb_treat_all,
        "treat_none": nb_treat_none,
    }


def binary_selective_metrics(
    confidences: np.ndarray,
    binary_labels: np.ndarray,
    *,
    coverage_points: Iterable[float],
) -> List[Dict[str, float]]:
    """Compute accuracy vs coverage curve for selective prediction."""

    idx_sorted = np.argsort(-confidences)
    confidences = confidences[idx_sorted]
    binary_labels = binary_labels[idx_sorted]
    cumulative_correct = np.cumsum(binary_labels)
    total = len(binary_labels)
    curve: List[Dict[str, float]] = []

    for coverage in coverage_points:
        coverage = float(np.clip(coverage, 0.0, 1.0))
        k = max(1, int(round(coverage * total)))
        accuracy = float(cumulative_correct[:k].sum() / k)
        curve.append({"coverage": coverage, "accuracy": accuracy})

    return curve
