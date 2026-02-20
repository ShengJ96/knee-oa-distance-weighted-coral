#!/usr/bin/env python3
"""Cross-domain evaluation: Test models on external datasets.

This script evaluates models trained on one dataset using another dataset.

External Validation Strategy:
- Set A trained model → evaluated on Set B FULL (train+val+test = 2136 samples)
- Set B trained model → evaluated on Set A test+val (2484 samples)

Usage:
    uv run python scripts/cross_domain_evaluation.py

Output:
    experiments/reports/predictions/cross_domain_*.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.pytorch_dataset import KneeOADataset
from src.training.cli import build_model, load_config


def get_predictions(model, dataloader, device):
    """Get predictions from model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle both dict format {"image": ..., "label": ...} and tuple format (image, label)
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].cpu().numpy()
            else:
                # Tuple format: (images, labels)
                images, labels = batch
                images = images.to(device)
                labels = labels.cpu().numpy()

            outputs = model(images)

            # Handle ordinal (CORAL) vs softmax outputs
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs

            # Check if ordinal output (cumulative probabilities)
            if logits.shape[-1] == 4:  # CORAL: 4 thresholds for 5 classes
                # Convert cumulative logits to class probabilities
                cum_probs = torch.sigmoid(logits)
                # P(Y=k) = P(Y>k-1) - P(Y>k)
                probs = torch.zeros(logits.shape[0], 5, device=device)
                probs[:, 0] = 1 - cum_probs[:, 0]
                for k in range(1, 4):
                    probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]
                probs[:, 4] = cum_probs[:, 3]
                preds = probs.argmax(dim=1).cpu().numpy()
                probs = probs.cpu().numpy()
            else:  # Softmax: 5 classes
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def load_external_test_set(dataset_name: str, img_size: int = 384, batch_size: int = 32):
    """Load external dataset for cross-domain validation.

    Strategy:
    - Set B: Use FULL dataset (train+val+test = 2136) because it's small
    - Set A: Use test+val only (2484) because it's already large enough
    """
    if dataset_name == "set_a":
        data_root = "dataset/set_a"
        splits_to_use = ["test", "val"]  # Set A is large, test+val is enough
    elif dataset_name == "set_b":
        data_root = "dataset/set_b"
        splits_to_use = ["train", "val", "test"]  # Set B is small, use ALL
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    from torch.utils.data import ConcatDataset

    datasets = []
    for split in splits_to_use:
        ds = KneeOADataset(
            data_root=data_root,
            split=split,
            target_size=(img_size, img_size),
            transform=None,
        )
        datasets.append(ds)
        print(f"  Loaded {split}: {len(ds)} samples")

    combined_dataset = ConcatDataset(datasets)
    print(f"  Total external validation: {len(combined_dataset)} samples")

    test_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


def evaluate_cross_domain(
    config_path: str,
    checkpoint_path: str,
    external_dataset: str,
    output_path: str,
    device: str = "cuda:0",
):
    """Evaluate a model on an external dataset."""
    print(f"\n{'='*60}")
    print(f"Cross-Domain Evaluation")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"External Dataset: {external_dataset}")
    print(f"{'='*60}\n")

    # Load config and model
    cfg = load_config(Path(config_path))
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    img_size = data_cfg.get("image_size", 384)
    batch_size = data_cfg.get("batch_size", 32)

    # Build and load model
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load external test set (with optional val set)
    test_loader = load_external_test_set(external_dataset, img_size, batch_size)
    print(f"External validation set size: {len(test_loader.dataset)}")

    # Get predictions
    predictions, labels, probabilities = get_predictions(model, test_loader, device)

    # Calculate metrics
    accuracy = (predictions == labels).mean()
    print(f"\nCross-Domain Accuracy: {accuracy*100:.2f}%")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        accuracy=accuracy,
        num_samples=len(labels),
        source_config=str(config_path),
        external_dataset=external_dataset,
    )

    print(f"Saved to: {output_path}")
    return accuracy


def main():
    """Run all cross-domain evaluations.

    External validation sample sizes:
    - Set A → Set B: FULL (train+val+test) = 2136 samples
    - Set B → Set A: test+val = 2484 samples
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow.")

    output_dir = Path("experiments/reports/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define cross-domain evaluation pairs
    # Format: (name, config, checkpoint_dir, external_dataset)
    evaluations = [
        # Set A trained → Test on Set B FULL (2136 samples)
        (
            "cross_seta_to_setb_baseline",
            "experiments/configs/stage6/ablation/stage6_siglip_set_a_baseline.yaml",
            "experiments/models/stage6/siglip_ord_coral/set_a_baseline",
            "set_b",
        ),
        (
            "cross_seta_to_setb_costsensitive",
            "experiments/configs/stage6/ablation/stage6_siglip_set_a_costsensitive.yaml",
            "experiments/models/stage6/siglip_ord_coral/set_a_costsensitive",
            "set_b",
        ),
        # Set B trained → Test on Set A test+val (2484 samples)
        (
            "cross_setb_to_seta_baseline",
            "experiments/configs/stage6/ablation/stage6_siglip_set_b_baseline.yaml",
            "experiments/models/stage6/siglip_ord_coral/set_b_baseline",
            "set_a",
        ),
        (
            "cross_setb_to_seta_costsensitive",
            "experiments/configs/stage6/ablation/stage6_siglip_set_b_costsensitive.yaml",
            "experiments/models/stage6/siglip_ord_coral/set_b_costsensitive",
            "set_a",
        ),
    ]

    results = []

    for name, config, ckpt_dir, ext_dataset in evaluations:
        # Find best checkpoint
        ckpt_dir = Path(ckpt_dir)
        checkpoints = list(ckpt_dir.glob("best_model_epoch_*.pth"))

        if not checkpoints:
            print(f"\nWARNING: No checkpoint found in {ckpt_dir}, skipping {name}")
            continue

        checkpoint = sorted(checkpoints)[-1]  # Latest best model
        output_path = output_dir / f"{name}.npz"

        try:
            acc = evaluate_cross_domain(
                config_path=config,
                checkpoint_path=str(checkpoint),
                external_dataset=ext_dataset,
                output_path=str(output_path),
                device=device,
            )
            results.append({"name": name, "accuracy": acc, "status": "success"})
        except Exception as e:
            print(f"ERROR evaluating {name}: {e}")
            results.append({"name": name, "accuracy": 0, "status": f"error: {e}"})

    # Print summary
    print("\n" + "="*60)
    print("Cross-Domain Evaluation Summary")
    print("="*60)

    for r in results:
        status = "✅" if r["status"] == "success" else "❌"
        print(f"{status} {r['name']}: {r['accuracy']*100:.2f}%")

    # Save summary
    summary_path = output_dir / "cross_domain_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
