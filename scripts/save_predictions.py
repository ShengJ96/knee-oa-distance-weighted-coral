"""Save model predictions for statistical testing.

This script loads trained models and saves per-sample predictions to disk.
Run this on GPU server where checkpoints are stored.

Usage:
    uv run python scripts/save_predictions.py \
        --config experiments/configs/stage6/ablation/stage6_siglip_set_ab_baseline.yaml \
        --checkpoint experiments/models/stage6/siglip_ord_coral/set_ab_baseline/best_model_epoch_*.pth \
        --device cuda:0 \
        --output experiments/reports/predictions/stage6_siglip_set_ab_baseline.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data.pytorch_dataset import KneeOADataset, get_data_transforms
from src.training.cli import build_model


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: Path, model, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle DDP checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def predict_dataset(model, dataloader, device, num_classes=5):
    """Run inference and collect predictions."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Handle different output types
            if hasattr(outputs, 'probs'):
                # OrdinalHeadOutput: already has probs computed
                probs = outputs.probs
            elif hasattr(outputs, 'logits'):
                # Has logits attribute
                logits = outputs.logits
                if logits.dim() == 2 and logits.size(1) == num_classes - 1:
                    # CORAL/CORN ordinal logits
                    cumulative_probs = torch.sigmoid(logits)
                    probs = torch.zeros(logits.size(0), num_classes, device=device)
                    probs[:, 0] = 1 - cumulative_probs[:, 0]
                    for k in range(1, num_classes - 1):
                        probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
                    probs[:, -1] = cumulative_probs[:, -1]
                else:
                    probs = torch.softmax(logits, dim=1)
            elif isinstance(outputs, dict):
                logits = outputs.get("ordinal_logits", outputs.get("logits"))
                if logits.dim() == 2 and logits.size(1) == num_classes - 1:
                    cumulative_probs = torch.sigmoid(logits)
                    probs = torch.zeros(logits.size(0), num_classes, device=device)
                    probs[:, 0] = 1 - cumulative_probs[:, 0]
                    for k in range(1, num_classes - 1):
                        probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
                    probs[:, -1] = cumulative_probs[:, -1]
                else:
                    probs = torch.softmax(logits, dim=1)
            else:
                # Raw tensor output (standard classification)
                logits = outputs
                probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = torch.device(args.device)

    # Create dataset using the correct API
    data_cfg = config["data"]
    target_size = tuple(data_cfg.get("target_size", [384, 384]))

    # Get transform for evaluation (no augmentation)
    transform = get_data_transforms(
        split=args.split,
        target_size=target_size,
        augment=False,
        medical_variant=data_cfg.get("medical_variant", "none"),
        augmentation_library="torchvision",
    )

    # Check for multi-source configuration
    multi_source = data_cfg.get("multi_source", {})
    if multi_source.get("enabled", False) and "roots" in multi_source:
        # Multi-source dataset: concatenate multiple data roots
        data_roots = multi_source["roots"]
        datasets = []
        for root in data_roots:
            ds = KneeOADataset(
                data_root=Path(root),
                split=args.split,
                transform=transform,
                target_size=target_size,
                limit_per_class=None,
            )
            datasets.append(ds)
        dataset = ConcatDataset(datasets)
        root_str = " + ".join(data_roots)
        print(f"Multi-source dataset: {root_str}")
    else:
        # Single-source dataset
        data_root = Path(data_cfg["root"])
        dataset = KneeOADataset(
            data_root=data_root,
            split=args.split,
            transform=transform,
            target_size=target_size,
            limit_per_class=None,
        )
        root_str = str(data_root)

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Dataset: {len(dataset)} samples from {root_str}/{args.split}")

    # Load model
    model_cfg = config["model"]
    num_classes = model_cfg["params"]["num_classes"]
    model = build_model(model_cfg).to(device)

    # Find and load checkpoint
    ckpt_path = Path(args.checkpoint)
    if "*" in args.checkpoint:
        matches = list(ckpt_path.parent.glob(ckpt_path.name))
        if not matches:
            raise FileNotFoundError(f"No checkpoint: {args.checkpoint}")
        ckpt_path = matches[0]

    model = load_checkpoint(ckpt_path, model, device)
    print(f"Loaded: {ckpt_path}")

    # Predict
    preds, labels, probs = predict_dataset(model, dataloader, device, num_classes)
    acc = (preds == labels).mean() * 100

    print(f"\nAccuracy: {acc:.2f}%")
    print(f"Samples: {len(labels)}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        predictions=preds,
        labels=labels,
        probabilities=probs,
        accuracy=acc,
        num_samples=len(labels)
    )

    # Save metadata
    meta = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_samples": len(labels),
        "accuracy": float(acc),
        "class_distribution": {
            "true": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "pred": {int(k): int(v) for k, v in zip(*np.unique(preds, return_counts=True))}
        }
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
