#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
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


def parse_targets(value: str) -> dict[str, int]:
    parts = [int(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Targets must be in 'train,val,test' format.")
    return {"train": parts[0], "val": parts[1], "test": parts[2]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean set_b by removing duplicates/conflicts and hit target split totals without moving samples."
    )
    parser.add_argument(
        "--source-root",
        default="dataset/set_b_backup_20260127_0023",
        help="Source set_b root to clean",
    )
    parser.add_argument(
        "--dest-root",
        default="dataset/set_b",
        help="Destination set_b root",
    )
    parser.add_argument(
        "--targets",
        default="1136,243,243",
        help="Target totals for train,val,test (e.g., 1136,243,243)",
    )
    parser.add_argument("--seed", type=int, default=202409, help="Shuffle seed")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (create new set_b).",
    )
    parser.add_argument(
        "--report",
        default="experiments/reports/set_b_clean_report_target.json",
        help="Report JSON path",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    targets = parse_targets(args.targets)
    rng = random.Random(args.seed)

    entries = scan_split_root(source_root)
    by_hash = defaultdict(list)
    for e in entries:
        by_hash[e.sha256].append(e)

    conflict_hashes = {h for h, items in by_hash.items() if len({e.label for e in items}) > 1}

    # Build option groups (only allow 1 or 2 splits per hash)
    options = defaultdict(list)
    hash_meta = {}
    for h, items in by_hash.items():
        if h in conflict_hashes:
            continue
        splits = tuple(sorted({e.split for e in items}))
        if len(splits) > 2:
            raise RuntimeError(f"Hash {h} appears in more than 2 splits: {splits}")
        options[splits].append(h)
        hash_meta[h] = {
            "label": items[0].label,
            "paths": defaultdict(list),
        }
        for e in items:
            hash_meta[h]["paths"][e.split].append(e.path)

    fixed_counts = Counter()
    for split in SPLITS:
        fixed_counts[split] = len(options[(split,)])

    remaining = {
        split: targets[split] - fixed_counts[split] for split in SPLITS
    }

    # Sanity checks
    total_hashes = sum(len(hs) for hs in options.values())
    if sum(targets.values()) != total_hashes:
        raise RuntimeError(
            f"Target totals {targets} do not sum to {total_hashes} unique hashes."
        )
    if any(v < 0 for v in remaining.values()):
        raise RuntimeError(f"Targets below fixed counts: fixed={fixed_counts}, targets={targets}")

    # Solve for pairwise groups: train-test, train-val, val-test
    n_tt = len(options[("test", "train")])
    n_tv = len(options[("train", "val")])
    n_vt = len(options[("test", "val")])

    r_train, r_val, r_test = remaining["train"], remaining["val"], remaining["test"]

    # x + y = r_train; (n_tv - y) + z = r_val; (n_tt - x) + (n_vt - z) = r_test
    # choose y in feasible range
    y_min = max(0, r_train - n_tt, n_tv - r_val)
    y_max = min(n_tv, r_train, n_tv - r_val + n_vt)
    if y_min > y_max:
        raise RuntimeError("No feasible assignment for the requested targets.")

    # choose mid-range y for stability
    y = (y_min + y_max) // 2
    x = r_train - y
    z = r_val - (n_tv - y)

    if not (0 <= x <= n_tt and 0 <= y <= n_tv and 0 <= z <= n_vt):
        raise RuntimeError("Computed x/y/z out of bounds.")

    if not args.apply:
        report = {
            "source_root": str(source_root),
            "dest_root": str(dest_root),
            "total_hashes": total_hashes,
            "conflict_hashes": len(conflict_hashes),
            "fixed_counts": dict(fixed_counts),
            "targets": targets,
            "remaining": remaining,
            "n_train_test": n_tt,
            "n_train_val": n_tv,
            "n_val_test": n_vt,
            "x_train_from_train_test": x,
            "y_train_from_train_val": y,
            "z_val_from_val_test": z,
        }
        print(json.dumps(report, indent=2))
        return

    # Prepare destination
    if dest_root.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_root = dest_root.parent / f"{dest_root.name}_backup_{timestamp}"
        os.rename(dest_root, backup_root)

    for split in SPLITS:
        for label in range(5):
            ensure_dir(dest_root / split / str(label))

    # Assign hashes for multi-split groups
    assignment = {}
    # fixed
    for split in SPLITS:
        for h in options[(split,)]:
            assignment[h] = split

    # train-test group
    tt_hashes = options[("test", "train")]
    rng.shuffle(tt_hashes)
    for h in tt_hashes[:x]:
        assignment[h] = "train"
    for h in tt_hashes[x:]:
        assignment[h] = "test"

    # train-val group
    tv_hashes = options[("train", "val")]
    rng.shuffle(tv_hashes)
    for h in tv_hashes[:y]:
        assignment[h] = "train"
    for h in tv_hashes[y:]:
        assignment[h] = "val"

    # val-test group
    vt_hashes = options[("test", "val")]
    rng.shuffle(vt_hashes)
    for h in vt_hashes[:z]:
        assignment[h] = "val"
    for h in vt_hashes[z:]:
        assignment[h] = "test"

    # Copy files into destination
    split_counts = {split: Counter() for split in SPLITS}
    for h, split in assignment.items():
        label = hash_meta[h]["label"]
        # pick a source path from the assigned split if available, else any path
        if hash_meta[h]["paths"].get(split):
            src = hash_meta[h]["paths"][split][0]
        else:
            src = next(iter(hash_meta[h]["paths"].values()))[0]
        dst = dest_root / split / str(label) / src.name
        safe_copy(src, dst)
        split_counts[split][label] += 1

    # Rebuild raw (one copy per hash, consensus only)
    raw_root = source_root / "raw"
    if raw_root.exists():
        raw_hashes = scan_raw_root(raw_root)
        raw_label_dirs = build_raw_label_map(raw_root)
        for label in range(5):
            ensure_dir(dest_root / "raw" / raw_label_dirs.get(label, f"grade{label}_clean"))
        for h, info in hash_meta.items():
            label = info["label"]
            src = None
            for p in raw_hashes.get(h, []):
                parent = p.parent.name
                if parent.startswith("grade") and int(parent.split("_")[0].replace("grade", "")) == label:
                    src = p
                    break
            if src is None:
                src = next(iter(info["paths"].values()))[0]
            raw_dir = dest_root / "raw" / raw_label_dirs.get(label, f"grade{label}_clean")
            safe_copy(src, raw_dir / src.name)

    report = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "total_hashes": total_hashes,
        "conflict_hashes": len(conflict_hashes),
        "fixed_counts": dict(fixed_counts),
        "targets": targets,
        "remaining": remaining,
        "n_train_test": n_tt,
        "n_train_val": n_tv,
        "n_val_test": n_vt,
        "x_train_from_train_test": x,
        "y_train_from_train_val": y,
        "z_val_from_val_test": z,
        "final_split_counts": {s: dict(split_counts[s]) for s in SPLITS},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
