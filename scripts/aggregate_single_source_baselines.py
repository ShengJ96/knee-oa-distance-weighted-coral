#!/usr/bin/env python3
"""Aggregate Stage 5 baseline results into JSON reports.

This utility scans experiment directories for `metadata.json` files created by
`knee-oa-train train`, filters to foundation models, and writes per-dataset
single-source summaries as well as a combined multi-source results file under
`experiments/reports/`.

Example:
  uv run python scripts/aggregate_single_source_baselines.py \\
    --roots experiments/models/foundation/general experiments/models/foundation/medical
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

FOUNDATION_REGISTRIES = {"foundation_general", "foundation_medical"}
REQUIRED_FILENAME = "metadata.json"


def discover_metadata(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield metadata files under the specified directories."""
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.name == REQUIRED_FILENAME:
            yield path
        elif path.is_dir():
            for candidate in path.rglob(REQUIRED_FILENAME):
                yield candidate


def normalize_dataset_name(root: str | None) -> str | None:
    if not root:
        return None
    path = Path(root)
    if path.name.startswith("set_"):
        return path.name.lower()
    for part in path.parts[::-1]:
        if part.startswith("set_"):
            return part.lower()
    return None


def _extract_single_dataset(data_cfg: Dict[str, object]) -> str | None:
    dataset_root = data_cfg.get("root")
    return normalize_dataset_name(str(dataset_root) if dataset_root else None)


def _extract_multi_datasets(data_cfg: Dict[str, object]) -> Tuple[str, ...] | None:
    multi_cfg = data_cfg.get("multi_source")
    if not isinstance(multi_cfg, dict) or not multi_cfg.get("enabled", True):
        return None

    dataset_roots: List[str] = []
    if isinstance(multi_cfg.get("datasets"), list):
        for spec in multi_cfg["datasets"]:
            if isinstance(spec, dict) and spec.get("root"):
                dataset_roots.append(str(spec["root"]))
    elif isinstance(multi_cfg.get("roots"), list):
        dataset_roots.extend(str(root) for root in multi_cfg["roots"] if root)

    dataset_names = [normalize_dataset_name(root) for root in dataset_roots]
    dataset_names = [name for name in dataset_names if name]
    if not dataset_names:
        return None

    unique_sorted = tuple(sorted({name.lower() for name in dataset_names}))
    return unique_sorted if unique_sorted else None


def extract_record(
    path: Path,
) -> Tuple[str, Tuple[str, ...] | str, Dict[str, object]] | None:
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_cfg: Dict[str, object] = metadata.get("model", {})  # type: ignore[assignment]
    registry = str(model_cfg.get("registry", "")).lower()
    if registry not in FOUNDATION_REGISTRIES:
        return None

    data_cfg: Dict[str, object] = metadata.get("data", {})  # type: ignore[assignment]
    dataset_name = _extract_single_dataset(data_cfg)
    multi_datasets = _extract_multi_datasets(data_cfg)

    if dataset_name:
        result_type: str = "single"
        key: Tuple[str, ...] | str = dataset_name
    elif multi_datasets:
        result_type = "multi"
        key = multi_datasets
    else:
        return None

    experiment_cfg: Dict[str, object] = metadata.get("experiment", {})  # type: ignore[assignment]
    results_cfg: Dict[str, object] = metadata.get("results", {})  # type: ignore[assignment]
    summary_cfg: Dict[str, object] = results_cfg.get("summary", {})  # type: ignore[assignment]

    record: Dict[str, object] = {
        "experiment_name": experiment_cfg.get("name"),
        "config": metadata.get("config"),
        "model_registry": registry,
        "model_key": model_cfg.get("key"),
        "dataset_root": data_cfg.get("root"),
        "output_dir": experiment_cfg.get("output_dir"),
        "figures_dir": experiment_cfg.get("figures_dir"),
        "reports_dir": experiment_cfg.get("reports_dir"),
        "best_epoch": summary_cfg.get("best_epoch"),
        "best_val_acc": summary_cfg.get("best_val_acc"),
        "test_metrics": results_cfg.get("test_metrics"),
        "best_model_path": results_cfg.get("best_model_path"),
        "metadata_path": str(path),
    }
    if isinstance(key, tuple):
        record["source_datasets"] = list(key)
    return result_type, key, record


def aggregate(
    metadata_files: Iterable[Path],
    *,
    datasets: set[str] | None = None,
) -> Tuple[
    Dict[str, List[Dict[str, object]]], Dict[Tuple[str, ...], List[Dict[str, object]]]
]:
    single_grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    multi_grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    seen_single: set[tuple[str, str]] = set()
    seen_multi: set[tuple[Tuple[str, ...], str]] = set()

    for meta_path in metadata_files:
        extracted = extract_record(meta_path)
        if not extracted:
            continue
        result_type, key, record = extracted
        if result_type == "single":
            dataset = str(key)
            if datasets and dataset not in datasets:
                continue
            dedupe_key = (dataset, str(record.get("experiment_name")))
            if dedupe_key in seen_single:
                continue
            seen_single.add(dedupe_key)
            single_grouped[dataset].append(record)
        else:
            combo = tuple(key)  # type: ignore[arg-type]
            dedupe_key = (combo, str(record.get("experiment_name")))
            if dedupe_key in seen_multi:
                continue
            seen_multi.add(dedupe_key)
            multi_grouped[combo].append(record)

    for dataset, records in single_grouped.items():
        records.sort(
            key=lambda rec: float(rec.get("best_val_acc") or float("-inf")),
            reverse=True,
        )
    for combo, records in multi_grouped.items():
        records.sort(
            key=lambda rec: float(rec.get("best_val_acc") or float("-inf")),
            reverse=True,
        )
    return single_grouped, multi_grouped


def write_reports(
    single_grouped: Dict[str, List[Dict[str, object]]],
    multi_grouped: Dict[Tuple[str, ...], List[Dict[str, object]]],
    *,
    output_dir: Path,
    overwrite: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for dataset, records in single_grouped.items():
        if not records:
            continue
        out_path = output_dir / f"{dataset}_single_domain.json"
        if out_path.exists() and not overwrite:
            print(f"⚠️ Skip writing {out_path} (exists). Use --overwrite to replace.")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        written.append(out_path)
        print(f"✓ Wrote {len(records)} records to {out_path}")

    if multi_grouped:
        combined: List[Dict[str, object]] = []
        for combo, records in multi_grouped.items():
            for record in records:
                combined.append(record)
        combined.sort(
            key=lambda rec: (
                tuple(rec.get("source_datasets", [])),
                -float(rec.get("best_val_acc") or float("-inf")),
            ),
        )
        multi_path = output_dir / "multi_source_baseline_results.json"
        if multi_path.exists() and not overwrite:
            print(f"⚠️ Skip writing {multi_path} (exists). Use --overwrite to replace.")
        else:
            with open(multi_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            written.append(multi_path)
            combos = {tuple(rec.get("source_datasets", [])) for rec in combined}
            print(
                f"✓ Wrote {len(combined)} multi-source records covering {len(combos)} combos to {multi_path}"
            )

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage 5 foundation baselines into summary JSON files."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["experiments/models"],
        help="Directories (or metadata files) to scan for experiment outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filters for single-source aggregation (e.g. set_a set_b).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/reports",
        help="Directory where aggregated reports will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.roots]
    dataset_filters = {ds.lower() for ds in args.datasets} if args.datasets else None

    metadata_files = list(discover_metadata(roots))
    if not metadata_files:
        print("⚠️ No metadata.json files found. Nothing to aggregate.")
        return

    single_grouped, multi_grouped = aggregate(metadata_files, datasets=dataset_filters)
    if not single_grouped and not multi_grouped:
        print("⚠️ No foundation model metadata matched the provided filters.")
        return

    write_reports(
        single_grouped,
        multi_grouped,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
