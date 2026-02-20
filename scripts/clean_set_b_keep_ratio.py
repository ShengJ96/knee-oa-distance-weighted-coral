#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Entry:
    split: str
    label: int
    path: Path
    sha256: str


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_split_root(root: Path) -> list[Entry]:
    entries: list[Entry] = []
    for split in SPLITS:
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
                            path=path,
                            sha256=sha256_path(path),
                        )
                    )
    return entries


def scan_raw_root(raw_root: Path) -> dict[str, list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in raw_root.rglob("*"):
        if path.is_file():
            by_hash[sha256_path(path)].append(path)
    return by_hash


def build_raw_label_map(raw_root: Path) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for grade_dir in raw_root.iterdir():
        if not grade_dir.is_dir():
            continue
        name = grade_dir.name
        if name.startswith("grade"):
            label = int(name.split("_")[0].replace("grade", ""))
            label_map[label] = name
    return label_map


def compute_targets(entries: list[Entry], conflict_hashes: set[str]) -> dict[int, dict[str, int]]:
    # counts of occurrences (not unique) per split/label, excluding conflict hashes
    occ = {split: Counter() for split in SPLITS}
    for e in entries:
        if e.sha256 in conflict_hashes:
            continue
        occ[e.split][e.label] += 1

    # total unique per label
    label_hashes = defaultdict(set)
    for e in entries:
        if e.sha256 in conflict_hashes:
            continue
        label_hashes[e.label].add(e.sha256)

    targets: dict[int, dict[str, int]] = {}
    for label, hashes in label_hashes.items():
        total_unique = len(hashes)
        split_counts = {split: occ[split][label] for split in SPLITS}
        total_occ = sum(split_counts.values()) or 1
        proportions = {split: split_counts[split] / total_occ for split in SPLITS}

        raw_targets = {split: proportions[split] * total_unique for split in SPLITS}
        base = {split: int(raw_targets[split]) for split in SPLITS}
        remainder = total_unique - sum(base.values())

        # distribute remainder by largest fractional parts
        frac_order = sorted(
            SPLITS, key=lambda s: (raw_targets[s] - base[s]), reverse=True
        )
        for split in frac_order[:remainder]:
            base[split] += 1

        targets[label] = base
    return targets


def assign_splits(
    entries: list[Entry], conflict_hashes: set[str], targets: dict[int, dict[str, int]]
) -> dict[str, str]:
    # hash -> {label, splits, paths}
    meta = {}
    for e in entries:
        if e.sha256 in conflict_hashes:
            continue
        if e.sha256 not in meta:
            meta[e.sha256] = {"label": e.label, "splits": set(), "paths": defaultdict(list)}
        meta[e.sha256]["splits"].add(e.split)
        meta[e.sha256]["paths"][e.split].append(e.path)

    current = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}

    # first assign hashes with a single candidate split
    singles = [h for h, info in meta.items() if len(info["splits"]) == 1]
    for h in singles:
        info = meta[h]
        split = next(iter(info["splits"]))
        assignment[h] = split
        current[split][info["label"]] += 1

    # assign multi-split hashes
    multis = [h for h, info in meta.items() if len(info["splits"]) > 1]
    for h in multis:
        info = meta[h]
        label = info["label"]
        candidates = sorted(info["splits"])
        # choose split with largest remaining need
        best_split = None
        best_remaining = None
        for split in candidates:
            remaining = targets[label][split] - current[split][label]
            if best_remaining is None or remaining > best_remaining:
                best_remaining = remaining
                best_split = split
        if best_split is None:
            best_split = candidates[0]
        assignment[h] = best_split
        current[best_split][label] += 1

    return assignment


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        for i in range(1, 1000):
            alt = dst.with_name(f"{stem}_dup{i}{suffix}")
            if not alt.exists():
                dst = alt
                break
    copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean set_b by removing duplicates and conflict hashes while preserving split ratios."
    )
    parser.add_argument("--root", default="dataset/set_b", help="Existing set_b root")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes: rename original and create cleaned set_b",
    )
    parser.add_argument(
        "--backup-name",
        default=None,
        help="Optional backup folder name (default: set_b_backup_YYYYMMDD_HHMM)",
    )
    parser.add_argument(
        "--report",
        default="experiments/reports/set_b_clean_report.json",
        help="Report JSON path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    entries = scan_split_root(root)
    by_hash = defaultdict(list)
    for e in entries:
        by_hash[e.sha256].append(e)

    conflict_hashes = {h for h, items in by_hash.items() if len({e.label for e in items}) > 1}

    targets = compute_targets(entries, conflict_hashes)
    assignment = assign_splits(entries, conflict_hashes, targets)

    # Summaries
    unique_hashes = len({e.sha256 for e in entries})
    kept_hashes = len(assignment)
    report = {
        "original_total_files": len(entries),
        "original_unique_hashes": unique_hashes,
        "conflict_hashes": len(conflict_hashes),
        "kept_unique_hashes": kept_hashes,
        "targets": targets,
    }

    if not args.apply:
        print(json.dumps(report, indent=2))
        return

    # rename original root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_name = args.backup_name or f"set_b_backup_{timestamp}"
    backup_root = root.parent / backup_name
    if backup_root.exists():
        raise RuntimeError(f"Backup path already exists: {backup_root}")
    os.rename(root, backup_root)

    # create new root structure
    for split in SPLITS:
        for label in range(5):
            ensure_dir(root / split / str(label))
    # copy split files
    # build hash -> label and split->paths mapping from backup
    backup_entries = scan_split_root(backup_root)
    meta = {}
    for e in backup_entries:
        if e.sha256 in conflict_hashes:
            continue
        if e.sha256 not in meta:
            meta[e.sha256] = {"label": e.label, "paths": defaultdict(list)}
        meta[e.sha256]["paths"][e.split].append(e.path)

    assigned_counts = {split: Counter() for split in SPLITS}
    for h, split in assignment.items():
        info = meta[h]
        label = info["label"]
        # choose a source path from the assigned split if available, else any path
        if info["paths"].get(split):
            src = info["paths"][split][0]
        else:
            src = next(iter(info["paths"].values()))[0]
        dst = root / split / str(label) / src.name
        safe_copy(src, dst)
        assigned_counts[split][label] += 1

    # rebuild raw (one copy per hash, consensus only)
    raw_root = backup_root / "raw"
    if raw_root.exists():
        raw_hashes = scan_raw_root(raw_root)
        raw_label_dirs = build_raw_label_map(raw_root)
        for label in range(5):
            ensure_dir(root / "raw" / raw_label_dirs.get(label, f"grade{label}_clean"))

        for h, info in meta.items():
            label = info["label"]
            # pick a source path from backup raw if available, else from split
            src = None
            for p in raw_hashes.get(h, []):
                parent = p.parent.name
                if parent.startswith("grade") and int(parent.split("_")[0].replace("grade", "")) == label:
                    src = p
                    break
            if src is None:
                # fallback to a split path
                src = next(iter(info["paths"].values()))[0]
            raw_dir = root / "raw" / raw_label_dirs.get(label, f"grade{label}_clean")
            ensure_dir(raw_dir)
            safe_copy(src, raw_dir / src.name)

    # write report
    final_counts = {split: dict(assigned_counts[split]) for split in SPLITS}
    report.update(
        {
            "final_split_counts": final_counts,
            "backup_root": str(backup_root),
        }
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
