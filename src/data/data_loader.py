"""Basic dataset indexing and exploration utilities for Knee OA project.

This module is lightweight and only depends on the standard library and
optionally Pillow/matplotlib when available. It is intended for Stage 1
data exploration and can be reused by notebooks and scripts.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KL_CLASS_NAMES: Dict[int, str] = {
    0: "normal",
    1: "doubtful",
    2: "minimal",
    3: "moderate",
    4: "severe",
}


def _list_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    if not dir_path.exists():
        return []
    return [p for p in dir_path.glob("*.*") if p.suffix.lower() in exts]


@dataclass
class SplitStats:
    split: str
    total: int
    by_grade: Dict[int, int]


class KneeOADataLoader:
    """Index and explore the dataset laid out as ``data_root/{split}/{0-4}/``.

    Parameters
    ---------
    root: str | Path
        Project root or dataset root. If a directory named "dataset" exists
        under `root`, it will be used automatically.
    seed: int
        Random seed used for sampling.
    """

    def __init__(self, root: os.PathLike | str = ".", seed: int = 42) -> None:
        root = Path(root)
        self.dataset_dir = root if (root / "0").exists() else root / "dataset"
        self.splits = ["train", "val", "test"]
        self._rng = random.Random(seed)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found at: {self.dataset_dir}. "
                "Expected structure: data_root/{train,val,test}/{0..4}/"
            )

    # ---------- Scanning ----------
    def scan_split(self, split: str) -> SplitStats:
        split_dir = self.dataset_dir / split
        by_grade: Dict[int, int] = {}
        total = 0
        for g in range(5):
            gdir = split_dir / str(g)
            count = len(_list_images(gdir))
            by_grade[g] = count
            total += count
        return SplitStats(split=split, total=total, by_grade=by_grade)

    def scan_all(self) -> Dict[str, SplitStats]:
        return {s: self.scan_split(s) for s in self.splits if (self.dataset_dir / s).exists()}

    # ---------- Sampling ----------
    def sample_images(
        self, split: str, grade: int, k: int = 4, shuffle: bool = True
    ) -> List[Path]:
        gdir = self.dataset_dir / split / str(grade)
        images = _list_images(gdir)
        if shuffle:
            self._rng.shuffle(images)
        return images[:k]

    # ---------- Quality checks ----------
    def verify_random_subset(
        self, split: str, per_grade: int = 10
    ) -> List[Tuple[Path, bool, Optional[str]]]:
        """Open a small random subset with PIL.Image.verify.

        Returns list of (path, ok, error_message_or_None).
        This is lightweight and safe to run even on large datasets.
        """
        try:
            from PIL import Image  # type: ignore
        except Exception:
            # Pillow not available; return empty verification results
            results: List[Tuple[Path, bool, Optional[str]]] = []
            for g in range(5):
                for p in self.sample_images(split, g, k=min(1, per_grade)):
                    results.append((p, False, "Pillow not installed"))
            return results

        results = []
        for g in range(5):
            imgs = self.sample_images(split, g, k=per_grade)
            for p in imgs:
                try:
                    with Image.open(p) as im:
                        im.verify()  # type: ignore[attr-defined]
                    results.append((p, True, None))
                except Exception as e:  # corrupt/unsupported
                    results.append((p, False, str(e)))
        return results

    # ---------- Convenience ----------
    @staticmethod
    def class_name(grade: int) -> str:
        return KL_CLASS_NAMES.get(grade, str(grade))

    def summary_table(self) -> List[Tuple[str, int, int, int, int, int, int]]:
        """Return a summary suitable for quick printing.

        Each row: (split, total, n0, n1, n2, n3, n4)
        """
        rows = []
        for split, stats in self.scan_all().items():
            n0 = stats.by_grade[0]
            n1 = stats.by_grade[1]
            n2 = stats.by_grade[2]
            n3 = stats.by_grade[3]
            n4 = stats.by_grade[4]
            rows.append((split, stats.total, n0, n1, n2, n3, n4))
        return rows
