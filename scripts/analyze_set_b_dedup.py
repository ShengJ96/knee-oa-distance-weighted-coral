#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    split: str
    label: int
    path: str
    sha256: str


RAW_LABEL_MAP = {
    "grade0_normal": 0,
    "grade1_doubtful": 1,
    "grade2_mild": 2,
    "grade3_moderate": 3,
    "grade4_severe": 4,
}


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_entries(root: Path) -> list[Entry]:
    entries: list[Entry] = []
    raw_root = root / "raw"
    for folder in RAW_LABEL_MAP:
        for path in sorted((raw_root / folder).glob("*")):
            if path.is_file():
                entries.append(
                    Entry(
                        split="raw",
                        label=RAW_LABEL_MAP[folder],
                        path=str(path),
                        sha256=sha256_path(path),
                    )
                )

    for split in ("train", "val", "test"):
        split_root = root / split
        for label_dir in sorted(split_root.glob("*")):
            if not label_dir.is_dir():
                continue
            label = int(label_dir.name)
            for path in sorted(label_dir.glob("*")):
                if path.is_file():
                    entries.append(
                        Entry(
                            split=split,
                            label=label,
                            path=str(path),
                            sha256=sha256_path(path),
                        )
                    )
    return entries


def summarize_by_hash(entries: list[Entry]) -> dict:
    by_hash: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_hash[entry.sha256].append(entry)

    duplicate_groups = [items for items in by_hash.values() if len(items) > 1]
    duplicate_files = sum(len(items) for items in duplicate_groups) - len(duplicate_groups)

    conflict_groups = []
    for items in duplicate_groups:
        labels = sorted({item.label for item in items})
        if len(labels) > 1:
            conflict_groups.append(
                {
                    "sha256": items[0].sha256,
                    "labels": labels,
                    "count": len(items),
                    "paths": [item.path for item in items],
                    "splits": sorted({item.split for item in items}),
                }
            )

    return {
        "total_files": len(entries),
        "unique_hashes": len(by_hash),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": duplicate_files,
        "label_conflict_groups": len(conflict_groups),
        "label_conflict_samples": conflict_groups[:25],
    }


def summarize_cross_split(entries: list[Entry]) -> dict:
    split_entries = [e for e in entries if e.split in {"train", "val", "test"}]
    by_hash: dict[str, list[Entry]] = defaultdict(list)
    for entry in split_entries:
        by_hash[entry.sha256].append(entry)

    cross_split = []
    for items in by_hash.values():
        splits = {item.split for item in items}
        if len(splits) > 1:
            cross_split.append(items)

    conflict_groups = []
    for items in cross_split:
        labels = sorted({item.label for item in items})
        if len(labels) > 1:
            conflict_groups.append(
                {
                    "sha256": items[0].sha256,
                    "labels": labels,
                    "count": len(items),
                    "paths": [item.path for item in items],
                    "splits": sorted({item.split for item in items}),
                }
            )

    return {
        "total_files": len(split_entries),
        "unique_hashes": len({e.sha256 for e in split_entries}),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicate_files": sum(len(items) for items in cross_split),
        "cross_split_label_conflicts": len(conflict_groups),
        "cross_split_conflict_samples": conflict_groups[:25],
    }


def write_csv(entries: list[Entry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "split", "label", "path"])
        for entry in entries:
            writer.writerow([entry.sha256, entry.split, entry.label, entry.path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Set B duplicates and label conflicts.")
    parser.add_argument(
        "--data-root",
        default="dataset/set_b",
        help="Path to dataset/set_b",
    )
    parser.add_argument(
        "--report",
        default="experiments/reports/set_b_dedup_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--csv",
        default="experiments/reports/set_b_hashes.csv",
        help="Output CSV listing of hashes",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    entries = collect_entries(root)
    raw_entries = [e for e in entries if e.split == "raw"]
    split_entries = [e for e in entries if e.split in {"train", "val", "test"}]

    report = {
        "data_root": str(root),
        "raw_summary": summarize_by_hash(raw_entries),
        "split_summary": summarize_by_hash(split_entries),
        "cross_split_summary": summarize_cross_split(entries),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    write_csv(entries, Path(args.csv))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
