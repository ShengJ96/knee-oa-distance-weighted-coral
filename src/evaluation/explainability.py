"""Explainability metric helpers (pointing game, insertion/deletion AUC)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _load_array(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    img = Image.open(path).convert("F")
    return np.array(img, dtype=np.float32)


def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0)
    max_v = arr.max()
    min_v = arr.min()
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = np.zeros_like(arr)
    return arr


def pointing_game(heatmap: np.ndarray, mask: np.ndarray) -> int:
    heatmap = normalize_heatmap(heatmap)
    mask = np.array(mask > 0, dtype=bool)
    if heatmap.size == 0 or mask.size == 0:
        return 0
    if heatmap.shape != mask.shape:
        raise ValueError("Heatmap and mask must share the same shape.")
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(mask[idx])


def pointing_game_score(paths: Sequence[Tuple[Path, Path]]) -> float:
    hits = 0
    for heatmap_path, mask_path in paths:
        heatmap = _load_array(heatmap_path)
        mask = _load_array(mask_path)
        hits += pointing_game(heatmap, mask)
    return hits / len(paths) if paths else 0.0


def insertion_deletion_auc(fractions: Sequence[float], scores: Sequence[float]) -> float:
    if len(fractions) != len(scores):
        raise ValueError("fractions and scores must be the same length")
    if not fractions:
        return 0.0
    fractions = np.asarray(fractions, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("fractions must be within [0, 1]")
    order = np.argsort(fractions)
    return float(np.trapezoid(scores[order], fractions[order]))


def load_curve_from_json(path: Path) -> Tuple[List[float], List[float]]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fractions = data.get("fractions") or data.get("coverage")
    scores = data.get("scores") or data.get("accuracy")
    if fractions is None or scores is None:
        raise ValueError(f"Curve JSON {path} must contain 'fractions'/'scores'.")
    return list(map(float, fractions)), list(map(float, scores))


def summarize_pointing_game(records: Iterable[int]) -> Dict[str, float]:
    records = list(records)
    total = len(records)
    hits = int(sum(records))
    return {"total": total, "hits": hits, "accuracy": (hits / total if total else 0.0)}
