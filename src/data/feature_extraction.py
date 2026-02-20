"""Feature extraction for traditional ML baselines.

This module provides a `MedicalImageFeatureExtractor` that turns an image into a
fixed-length 1D feature vector using lightweight statistics and textures. It is
robust to missing optional libraries: when scikit-image is unavailable, it will
fallback to a reduced feature set so that basic flows can still run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import stats


def _safe_import_skimage():
    try:
        from skimage import feature, filters
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte
        from skimage.feature import graycomatrix, graycoprops

        return {
            "feature": feature,
            "filters": filters,
            "rgb2gray": rgb2gray,
            "img_as_ubyte": img_as_ubyte,
            "graycomatrix": graycomatrix,
            "graycoprops": graycoprops,
        }
    except Exception:
        return None


def _read_grayscale(path: Path, resize: Optional[int] = 256) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("L")
    if resize is not None:
        # Keep aspect by thumbnail; then pad/crop to square center
        img = img.resize((resize, resize))
    arr = np.asarray(img, dtype=np.float32)
    return arr


@dataclass
class ExtractConfig:
    resize: Optional[int] = 256
    lbp_radius: int = 2
    lbp_points: int = 16
    glcm_distances: Tuple[int, ...] = (1, 2)
    glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    hist_bins: int = 32


class MedicalImageFeatureExtractor:
    """Extracts hand-crafted features for knee X-ray classification.

    Features (if skimage available):
      - Intensity stats: mean/std/skew/kurtosis + percentiles (10/50/90)
      - Histogram: 32-bin normalized grayscale histogram
      - Texture: LBP histogram (uniform)
      - GLCM props: contrast/homogeneity/energy/correlation/ASM averaged
      - Edges: Sobel magnitude stats

    Fallback (without skimage):
      - Intensity stats + histogram only
    """

    def __init__(self, cfg: Optional[ExtractConfig] = None) -> None:
        self.cfg = cfg or ExtractConfig()
        self._sk = _safe_import_skimage()

    # ---------- Core per-image extraction ----------
    def extract(self, path: Path) -> np.ndarray:
        x = _read_grayscale(path, self.cfg.resize)
        x_u8 = np.clip(x, 0, 255).astype(np.uint8)

        feats: List[float] = []

        # Intensity stats
        feats.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(stats.skew(x.reshape(-1))),
                float(stats.kurtosis(x.reshape(-1))),
                float(np.percentile(x, 10)),
                float(np.percentile(x, 50)),
                float(np.percentile(x, 90)),
            ]
        )

        # Histogram
        hist, _ = np.histogram(x_u8, bins=self.cfg.hist_bins, range=(0, 255), density=True)
        feats.extend(hist.astype(float))

        if self._sk is None:
            return np.asarray(feats, dtype=np.float32)

        feature = self._sk["feature"]
        filters = self._sk["filters"]
        graycomatrix = self._sk["graycomatrix"]
        graycoprops = self._sk["graycoprops"]

        # LBP (uniform)
        lbp = feature.local_binary_pattern(
            x_u8,
            P=self.cfg.lbp_points,
            R=self.cfg.lbp_radius,
            method="uniform",
        )
        n_bins = int(self.cfg.lbp_points + 2)
        lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        feats.extend(lbp_hist.astype(float))

        # GLCM
        glcm = graycomatrix(
            x_u8,
            distances=self.cfg.glcm_distances,
            angles=self.cfg.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
        props = ["contrast", "homogeneity", "energy", "correlation", "ASM"]
        for p in props:
            vals = graycoprops(glcm, p)
            feats.append(float(np.mean(vals)))

        # Sobel edges
        sob = filters.sobel(x)
        feats.extend(
            [
                float(np.mean(sob)),
                float(np.std(sob)),
                float(np.percentile(sob, 90)),
            ]
        )

        return np.asarray(feats, dtype=np.float32)

    # ---------- Batch helpers ----------
    def build_split(
        self,
        dataset_root: Path,
        split: str,
        limit_per_class: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
        """Build X, y from a split directory ``data_root/{split}/{0..4}/``.

        Parameters
        ---------
        limit_per_class: if set, cap samples per grade to at most this number.
        """
        split_dir = Path(dataset_root) / split
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        p_list: List[Path] = []

        rng = np.random.default_rng(42)
        for g in range(5):
            gdir = split_dir / str(g)
            paths = [p for p in gdir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if shuffle:
                rng.shuffle(paths)
            if limit_per_class is not None:
                paths = paths[: limit_per_class]
            for p in paths:
                try:
                    X_list.append(self.extract(p))
                    y_list.append(g)
                    p_list.append(p)
                except Exception:
                    # Skip unreadable files
                    continue

        X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int64)
        return X, y, p_list
