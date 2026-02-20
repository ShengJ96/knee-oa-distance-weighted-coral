#!/usr/bin/env python3
"""Compute dataset-level statistics (class counts, split counts, image sizes)."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_image_files(directory: Path) -> List[Path]:
    return [p for p in directory.glob("**/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def count_split(root: Path, split: str) -> Tuple[Dict[str, int], int]:
    split_dir = root / split
    counts: Dict[str, int] = {}
    total = 0
    if not split_dir.exists():
        return counts, total

    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        # fallback: treat split root as flat files
        files = list_image_files(split_dir)
        return {"all": len(files)}, len(files)

    for class_dir in class_dirs:
        files = list_image_files(class_dir)
        counts[class_dir.name] = len(files)
        total += len(files)
    return counts, total


def sample_image_sizes(files: List[Path], sample_size: int, rng: random.Random) -> Dict[str, float]:
    if not files:
        return {"count": 0}
    if sample_size and len(files) > sample_size:
        files = rng.sample(files, sample_size)

    widths: List[int] = []
    heights: List[int] = []
    for path in files:
        try:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception:
            continue

    if not widths:
        return {"count": 0}
    stats = {
        "count": len(widths),
        "mean_width": float(sum(widths) / len(widths)),
        "mean_height": float(sum(heights) / len(heights)),
        "min_width": float(min(widths)),
        "max_width": float(max(widths)),
        "min_height": float(min(heights)),
        "max_height": float(max(heights)),
    }
    return stats


def analyze_dataset(root: Path, splits: Iterable[str], sample_size: int, rng: random.Random) -> Dict:
    root = root.resolve()
    dataset_label = root.name
    summary = {"root": str(root), "splits": {}, "totals": {}}

    all_images: List[Path] = []
    total_images = 0
    for split in splits:
        per_class, total = count_split(root, split)
        summary["splits"][split] = {"per_class": per_class, "total": total}
        total_images += total
        for class_name, count in per_class.items():
            summary["totals"].setdefault(class_name, 0)
            summary["totals"][class_name] += count
        split_dir = root / split
        if split_dir.exists():
            all_images.extend(list_image_files(split_dir))

    summary["total_images"] = total_images
    summary["image_size_stats"] = sample_image_sizes(all_images, sample_size, rng)
    summary["label"] = dataset_label
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for KL grading datasets.")
    parser.add_argument("roots", nargs="+", type=Path, help="Dataset root directories (e.g., dataset/set_a).")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Dataset splits to scan.")
    parser.add_argument("--sample-size", type=int, default=200, help="Images to sample for size stats（0=all）。")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/reports/dataset_statistics.json"),
        help="Output JSON path。",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = {
        "args": {
            "roots": [str(r) for r in args.roots],
            "splits": args.splits,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        "datasets": {},
    }

    for root in args.roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root {root} 不存在")
        summary = analyze_dataset(root, args.splits, args.sample_size, rng)
        results["datasets"][summary["label"]] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Dataset statistics saved to {args.output}")


if __name__ == "__main__":
    main()
