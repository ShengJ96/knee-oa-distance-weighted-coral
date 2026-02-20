#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    expert: str
    label: int
    path: str
    sha256: str


LABEL_MAP = {
    "0Normal": 0,
    "1Doubtful": 1,
    "2Mild": 2,
    "3Moderate": 3,
    "4Severe": 4,
}


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_entries(expert_name: str, root: Path) -> list[Entry]:
    entries: list[Entry] = []
    for folder, label in LABEL_MAP.items():
        for path in sorted((root / folder).glob("*")):
            if path.is_file():
                entries.append(
                    Entry(
                        expert=expert_name,
                        label=label,
                        path=str(path),
                        sha256=sha256_path(path),
                    )
                )
    return entries


def summarize_expert(entries: list[Entry], expert: str) -> dict:
    expert_entries = [e for e in entries if e.expert == expert]
    label_counts = Counter(e.label for e in expert_entries)
    by_hash = defaultdict(list)
    for e in expert_entries:
        by_hash[e.sha256].append(e)

    duplicate_groups = [items for items in by_hash.values() if len(items) > 1]
    conflict_groups = []
    for items in duplicate_groups:
        labels = sorted({e.label for e in items})
        if len(labels) > 1:
            conflict_groups.append(
                {
                    "sha256": items[0].sha256,
                    "labels": labels,
                    "count": len(items),
                    "paths": [e.path for e in items],
                }
            )

    return {
        "total_files": len(expert_entries),
        "label_counts": dict(label_counts),
        "unique_hashes": len(by_hash),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": sum(len(items) for items in duplicate_groups) - len(duplicate_groups),
        "label_conflict_groups": len(conflict_groups),
        "label_conflict_samples": conflict_groups[:25],
    }


def summarize_cross_expert(entries: list[Entry]) -> dict:
    by_hash: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_hash[entry.sha256].append(entry)

    cross_groups = []
    conflict_groups = []
    for items in by_hash.values():
        experts = {e.expert for e in items}
        if len(experts) > 1:
            cross_groups.append(items)
            labels = sorted({e.label for e in items})
            if len(labels) > 1:
                conflict_groups.append(
                    {
                        "sha256": items[0].sha256,
                        "labels": labels,
                        "count": len(items),
                        "paths": [e.path for e in items],
                    }
                )

    # consensus distribution (agreed only)
    agreed_labels = []
    for items in cross_groups:
        labels = {e.label for e in items}
        if len(labels) == 1:
            agreed_labels.append(next(iter(labels)))
    agreed_counts = Counter(agreed_labels)

    # resolve conflicts by taking max label (ties go higher)
    resolved_labels = []
    for items in cross_groups:
        labels = {e.label for e in items}
        resolved_labels.append(max(labels))
    resolved_counts = Counter(resolved_labels)

    return {
        "unique_hashes": len(by_hash),
        "cross_expert_groups": len(cross_groups),
        "cross_expert_label_conflicts": len(conflict_groups),
        "cross_expert_conflict_samples": conflict_groups[:25],
        "consensus_only_label_counts": dict(agreed_counts),
        "consensus_only_total": sum(agreed_counts.values()),
        "resolved_max_label_counts": dict(resolved_counts),
        "resolved_max_total": sum(resolved_counts.values()),
    }


def write_csv(entries: list[Entry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "expert", "label", "path"])
        for entry in entries:
            writer.writerow([entry.sha256, entry.expert, entry.label, entry.path])


def write_conflicts(entries: list[Entry], output_path: Path) -> None:
    by_hash: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_hash[entry.sha256].append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "expert", "label", "path"])
        for h, items in sorted(by_hash.items()):
            labels = {e.label for e in items}
            if len(labels) > 1:
                for e in items:
                    writer.writerow([h, e.expert, e.label, e.path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw Set B expert folders.")
    parser.add_argument("--expert-a", default="MedicalExpert-I", help="Path to MedicalExpert-I")
    parser.add_argument("--expert-b", default="MedicalExpert-II", help="Path to MedicalExpert-II")
    parser.add_argument(
        "--report",
        default="experiments/reports/set_b_expert_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--csv",
        default="experiments/reports/set_b_expert_hashes.csv",
        help="Output CSV listing of hashes",
    )
    parser.add_argument(
        "--conflicts",
        default="experiments/reports/set_b_expert_conflicts.csv",
        help="Output CSV of cross-expert label conflicts",
    )
    args = parser.parse_args()

    expert_a = Path(args.expert_a)
    expert_b = Path(args.expert_b)

    entries = []
    entries.extend(collect_entries(expert_a.name, expert_a))
    entries.extend(collect_entries(expert_b.name, expert_b))

    report = {
        "expert_a": expert_a.name,
        "expert_b": expert_b.name,
        "total_files": len(entries),
        "expert_a_summary": summarize_expert(entries, expert_a.name),
        "expert_b_summary": summarize_expert(entries, expert_b.name),
        "cross_expert_summary": summarize_cross_expert(entries),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    write_csv(entries, Path(args.csv))
    write_conflicts(entries, Path(args.conflicts))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
