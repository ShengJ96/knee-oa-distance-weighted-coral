"""Baseline experiment runner using classical ML over hand-crafted features."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.feature_extraction import MedicalImageFeatureExtractor
from src.models.baseline.classical import BaselineClassifiers
from src.evaluation.metrics import classification_metrics, confusion_and_report


def run(
    dataset_root: Path | str = "dataset/set_a",
    output_dir: Path | str = "experiments/results/baseline",
    train_split: str = "train",
    val_split: str = "val",
    limit_per_class: int | None = 400,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = MedicalImageFeatureExtractor()

    # Build datasets
    X_train, y_train, _ = extractor.build_split(dataset_root, train_split, limit_per_class)
    X_val, y_val, _ = extractor.build_split(dataset_root, val_split, None)
    if X_train.size == 0 or X_val.size == 0:
        raise RuntimeError("Empty features. Check dataset structure and file formats.")

    # Train/eval models
    models = BaselineClassifiers.registry(random_state=random_state)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        res = BaselineClassifiers.fit_and_eval(name, model, X_train, y_train, X_val, y_val)
        results[name] = res.metrics

        # Save predictions and report
        y_pred = model.predict(X_val)
        cm, rep = confusion_and_report(y_val, y_pred)
        (output_dir / f"{name}_cm.npy").write_bytes(cm.astype(np.int32).tobytes())
        with open(output_dir / f"{name}_report.txt", "w") as f:
            f.write(rep)

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
