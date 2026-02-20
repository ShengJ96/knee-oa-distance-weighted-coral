#!/usr/bin/env python3
"""Convenience script to run Stage 2 classical baselines.

Usage:
  uv run python scripts/run_baseline_experiment.py --data dataset/set_a --out experiments/results/baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical ML baselines")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset",
        help="Dataset root directory (e.g. dataset/set_a)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/results/baseline",
        help="Output directory for baseline metrics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Max samples per class for training split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.experiments import baseline_experiments as be

    print("🔬 Running baseline experiments...")
    results = be.run(
        dataset_root=Path(args.data),
        output_dir=Path(args.out),
        limit_per_class=args.limit,
    )
    print("✅ Done. Results:")
    for key, value in results.items():
        print(key, value)


if __name__ == "__main__":
    main()

