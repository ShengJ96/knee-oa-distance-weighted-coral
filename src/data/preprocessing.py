"""Medical image preprocessing utilities for knee OA classification.

Provides reusable preprocessing functions for 2D PNG knee images:
- CLAHE contrast enhancement on grayscale with RGB mapping
- Center pad to square while keeping aspect ratio
- Preprocess factory with A/B/C presets for experiments

Dependencies: scikit-image (preferred). Falls back gracefully if missing.
"""

from __future__ import annotations

from typing import Callable, Optional
import os
import warnings
from PIL import Image, ImageOps

# Expose a version tag for notebooks to verify reloads
MONAI_AUG_VERSION = "2025-09-11-seq-no-compose-seeded"


# --- Lightweight classical imaging helpers (PIL/numpy only) ---
def _percentile_window(
    img: Image.Image, low_p: float = 2.0, high_p: float = 98.0
) -> Image.Image:
    """Percentile windowing on grayscale, then map back to RGB.

    Robustly suppress outliers and standardize contrast without external deps.
    """
    try:
        import numpy as np

        g = img.convert("L")
        a = np.asarray(g, dtype=np.float32)
        lo = float(np.percentile(a, low_p))
        hi = float(np.percentile(a, high_p))
        if hi <= lo:
            return g.convert("RGB")
        a = (a - lo) / (hi - lo)
        a = a.clip(0.0, 1.0)
        out = (a * 255.0).round().astype("uint8")
        return Image.fromarray(out, mode="L").convert("RGB")
    except Exception:
        return img.convert("L").convert("RGB")


def _unsharp_mask(
    img: Image.Image, radius: int = 2, amount: float = 1.0
) -> Image.Image:
    """Simple unsharp mask using PIL filters (edge enhancement for bone edges)."""
    try:
        from PIL import ImageFilter, ImageChops

        # Work in luminance to avoid color artifacts
        lum = img.convert("L")
        blurred = lum.filter(ImageFilter.GaussianBlur(radius=radius))
        # amount in [0,2] typically
        mask = ImageChops.subtract(lum, blurred)
        enhanced = ImageChops.add(lum, mask, scale=1.0, offset=0)
        if amount != 1.0:
            enhanced = ImageChops.blend(lum, enhanced, alpha=max(0.0, min(2.0, amount)))
        return Image.merge("RGB", (enhanced, enhanced, enhanced))
    except Exception:
        return img


def _auto_knee_roi_crop(img: Image.Image, margin: float = 0.08) -> Image.Image:
    """Heuristic knee ROI crop from a radiograph (2D X-ray).

    Steps:
      1) CLAHE (or equalize) on grayscale to amplify bone contrast.
      2) High threshold to get bright bone mask; keep largest connected region.
      3) Compute tight bbox, expand by `margin`, crop and return RGB image.

    Falls back to original image if anything fails (robust in dataloaders).
    """
    try:
        import numpy as np
        from skimage import measure

        g = img.convert("L")
        # Boost contrast first for robust masking
        g_clahe_rgb = _clahe_grayscale_to_rgb(g)
        g2 = g_clahe_rgb.convert("L")
        a = np.asarray(g2, dtype=np.float32)

        # High threshold at ~85th percentile to focus on dense bone regions
        thr = float(np.percentile(a, 85.0))
        mask = (a >= thr).astype(np.uint8)

        # Connected components, pick the largest
        lbl = measure.label(mask, connectivity=2)
        if lbl.max() == 0:
            return img
        # area per label
        areas = np.bincount(lbl.ravel())
        areas[0] = 0
        k = int(areas.argmax())
        rr, cc = np.where(lbl == k)
        if rr.size == 0:
            return img
        rmin, rmax = int(rr.min()), int(rr.max())
        cmin, cmax = int(cc.min()), int(cc.max())

        # Expand bbox by margin
        h, w = a.shape
        dh = int((rmax - rmin + 1) * margin)
        dw = int((cmax - cmin + 1) * margin)
        rmin = max(0, rmin - dh)
        rmax = min(h - 1, rmax + dh)
        cmin = max(0, cmin - dw)
        cmax = min(w - 1, cmax + dw)

        # Crop and return as RGB
        crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        return crop.convert("RGB")
    except Exception:
        return img


def _safe_knee_roi_crop(
    img: Image.Image,
    margin: float = 0.08,
    min_area_frac: float = 0.30,
    max_area_frac: float = 0.90,
    min_ar: float = 0.6,
    max_ar: float = 1.6,
    center_tol: float = 0.20,
) -> Image.Image:
    """Safer ROI crop with constraints; fallback to original if suspicious.

    Constraints relative to original image size:
    - area fraction in [min_area_frac, max_area_frac]
    - aspect ratio in [min_ar, max_ar]
    - crop center within image center ± center_tol (as fraction of width/height)
    """
    try:
        import numpy as np
        from skimage import measure

        g = img.convert("L")
        # Boost contrast first for robust masking
        g_clahe_rgb = _clahe_grayscale_to_rgb(g)
        g2 = g_clahe_rgb.convert("L")
        a = np.asarray(g2, dtype=np.float32)

        thr = float(np.percentile(a, 85.0))
        mask = (a >= thr).astype(np.uint8)
        lbl = measure.label(mask, connectivity=2)
        if lbl.max() == 0:
            return img

        areas = np.bincount(lbl.ravel())
        areas[0] = 0
        k = int(areas.argmax())
        rr, cc = np.where(lbl == k)
        if rr.size == 0:
            return img
        rmin, rmax = int(rr.min()), int(rr.max())
        cmin, cmax = int(cc.min()), int(cc.max())

        h, w = a.shape
        dh = int((rmax - rmin + 1) * margin)
        dw = int((cmax - cmin + 1) * margin)
        rmin = max(0, rmin - dh)
        rmax = min(h - 1, rmax + dh)
        cmin = max(0, cmin - dw)
        cmax = min(w - 1, cmax + dw)

        crop_w = (cmax + 1) - cmin
        crop_h = (rmax + 1) - rmin
        area_frac = (crop_w * crop_h) / float(w * h)
        ar = crop_w / max(1e-6, crop_h)
        cx = (cmin + cmax + 1) / 2.0
        cy = (rmin + rmax + 1) / 2.0
        cx_rel = abs(cx / w - 0.5)
        cy_rel = abs(cy / h - 0.5)

        if not (min_area_frac <= area_frac <= max_area_frac):
            return img
        if not (min_ar <= ar <= max_ar):
            return img
        if not (cx_rel <= center_tol and cy_rel <= center_tol):
            return img

        crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        return crop.convert("RGB")
    except Exception:
        return img


def _clahe_grayscale_to_rgb(
    image: Image.Image, clip_limit: float = 0.02
) -> Image.Image:
    """Apply CLAHE on grayscale and convert back to RGB.

    If scikit-image is unavailable, falls back to PIL equalize (global HE).
    """
    try:
        import numpy as np
        from skimage import exposure

        gray = image.convert("L")
        np_gray = np.array(gray)
        clahe = exposure.equalize_adapthist(np_gray, clip_limit=clip_limit)
        clahe_uint8 = (clahe * 255).astype("uint8")
        return Image.fromarray(clahe_uint8, mode="L").convert("RGB")
    except Exception:
        # Fallback: PIL global histogram equalization (weaker than CLAHE)
        return ImageOps.equalize(image.convert("L")).convert("RGB")


def _pad_to_square_keep_aspect(image: Image.Image, fill: int = 0) -> Image.Image:
    """Pad image to square canvas while keeping aspect ratio (centered)."""
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    delta_w = size - width
    delta_h = size - height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    return ImageOps.expand(image, padding, fill=fill)


def _monai_medical_augment(image: Image.Image) -> Image.Image:
    """Apply MONAI medical augmentation transforms to PIL Image.

    Pipeline: PIL -> numpy (C,H,W in [0,1]) -> MONAI -> numpy -> PIL.

    Notes:
    - Avoid `cache_grid=True` when `spatial_size` is dynamic; it warns and can
      trigger downstream issues. We specify `spatial_size` explicitly to be stable.
    - Handle MONAI returning either numpy arrays or torch tensors.
    """
    try:
        import numpy as np

        # Silence MONAI's pkg_resources deprecation warning (harmless but noisy)
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
        from monai.transforms import (
            RandAffine,
            RandGaussianNoise,
            RandAdjustContrast,
        )

        # Convert PIL -> grayscale -> float32 [0,1] and add channel dim
        img_array = np.array(image.convert("L"), dtype=np.float32) / 255.0  # (H, W)
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)

        # Stable affine settings: explicit spatial_size and no cache_grid
        h, w = int(img_array.shape[-2]), int(img_array.shape[-1])
        spatial_size = (h, w)

        # MONAI uses radians for rotate_range; 5 degrees ~= 0.0873 rad
        rot = (-0.0873, 0.0873)

        # Apply transforms sequentially to avoid Compose's set_random_state bug
        t1 = RandAffine(
            prob=0.3,
            rotate_range=rot,
            translate_range=(5.0, 5.0),
            scale_range=(0.95, 1.05),
            padding_mode="zeros",
            mode="bilinear",
            spatial_size=spatial_size,
        )
        t2 = RandGaussianNoise(prob=0.2, std=0.005)
        t3 = RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1))

        # Seed transforms with safe 32-bit values to avoid uint32 edge case
        try:
            max_seed = (1 << 32) - 2  # strictly below 2**32 - 1 to be safe
            base = int.from_bytes(os.urandom(4), "little") % max_seed
            t1.set_random_state(base)
            t2.set_random_state((base + 1) % max_seed)
            t3.set_random_state((base + 2) % max_seed)
        except Exception:
            pass

        result = t1(img_array)
        result = t2(result)
        result = t3(result)

        # Convert MONAI output back to numpy (H, W) safely
        try:
            import torch  # type: ignore

            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
        except Exception:
            pass

        # result is numpy with shape (1, H, W)
        if result.ndim == 3 and result.shape[0] == 1:
            result = result.squeeze(0)

        # Clip to [0,1] and convert to uint8 safely
        result = np.clip(result, 0.0, 1.0)
        result_uint8 = (result * 255.0).round().astype(np.uint8)

        return Image.fromarray(result_uint8, mode="L").convert("RGB")

    except Exception as e:
        print(f"Warning: MONAI augmentation failed: {e}, using original image")
        return image


# Global preprocessing functions (pickleable)
def preprocess_none(img: Image.Image) -> Image.Image:
    """No preprocessing."""
    return img


def preprocess_clahe_only(img: Image.Image) -> Image.Image:
    """CLAHE on grayscale, mapped back to RGB."""
    return _clahe_grayscale_to_rgb(img)


def preprocess_clahe_center_square(img: Image.Image) -> Image.Image:
    """Pad to square (center), then CLAHE."""
    img_sq = _pad_to_square_keep_aspect(img)
    return _clahe_grayscale_to_rgb(img_sq)


def preprocess_center_square_only(img: Image.Image) -> Image.Image:
    """Only pad to square."""
    return _pad_to_square_keep_aspect(img)


def preprocess_monai_medical_aug(img: Image.Image) -> Image.Image:
    """MONAI medical augmentation."""
    return _monai_medical_augment(img)


def preprocess_knee_roi_clahe_unsharp(img: Image.Image) -> Image.Image:
    """ROI-focused preprocessing for knee X-ray:
    1) Auto knee ROI crop
    2) Percentile windowing (2–98%)
    3) CLAHE (perceptual boost)
    4) Unsharp mask (edge emphasis)

    All steps are best-effort with safe fallbacks, returning RGB.
    """
    try:
        # 1) ROI crop
        x = _auto_knee_roi_crop(img)
        # 2) Percentile windowing
        x = _percentile_window(x, 2.0, 98.0)
        # 3) CLAHE on luminance
        x = _clahe_grayscale_to_rgb(x)
        # 4) Unsharp
        x = _unsharp_mask(x, radius=2, amount=1.0)
        return x
    except Exception:
        return _clahe_grayscale_to_rgb(img)


def preprocess_knee_roi_clahe_unsharp_v2(img: Image.Image) -> Image.Image:
    """Safer ROI + milder contrast for knee X-ray (recommended D'):
    1) Safe ROI crop with constraints (area/aspect/center), else fallback original
    2) Percentile windowing (2–98%) for outlier suppression
    3) CLAHE with lower clip limit (0.01) to avoid over-enhancement
    4) Mild unsharp (radius=1, amount=0.6) or skip if undesired
    """
    try:
        x = _safe_knee_roi_crop(
            img,
            margin=0.08,
            min_area_frac=0.30,
            max_area_frac=0.90,
            min_ar=0.6,
            max_ar=1.6,
            center_tol=0.20,
        )
        x = _percentile_window(x, 2.0, 98.0)
        x = _clahe_grayscale_to_rgb(x, clip_limit=0.01)
        x = _unsharp_mask(x, radius=1, amount=0.6)
        return x
    except Exception:
        return _clahe_grayscale_to_rgb(img)


def medical_preprocess_factory(
    variant: str = "none",
) -> Optional[Callable[[Image.Image], Image.Image]]:
    """Create a medical preprocessing function by name.

    Variants:
    - none: no-op
    - clahe_only: CLAHE on grayscale, mapped back to RGB
    - clahe_center_square: pad to square (center), then CLAHE
    - center_square_only: only pad to square
    - monai_medical_aug: MONAI medical augmentation (RandAffine + noise + contrast)
    """
    variant = (variant or "none").lower()

    if variant == "none":
        return preprocess_none
    elif variant in {"clahe_only", "clahe"}:
        return preprocess_clahe_only
    elif variant == "clahe_center_square":
        return preprocess_clahe_center_square
    elif variant == "center_square_only":
        return preprocess_center_square_only
    elif variant == "monai_medical_aug":
        return preprocess_monai_medical_aug
    elif variant in {"knee_roi", "knee_roi_clahe_unsharp", "roi_clahe_unsharp"}:
        return preprocess_knee_roi_clahe_unsharp
    elif variant in {"knee_roi_clahe_unsharp_v2", "knee_roi_safe"}:
        return preprocess_knee_roi_clahe_unsharp_v2
    else:
        # Unknown variant -> no-op
        return preprocess_none


__all__ = [
    "medical_preprocess_factory",
    "preprocess_none",
    "preprocess_clahe_only",
    "preprocess_clahe_center_square",
    "preprocess_center_square_only",
    "preprocess_monai_medical_aug",
    "preprocess_knee_roi_clahe_unsharp",
    "preprocess_knee_roi_clahe_unsharp_v2",
    "MONAI_AUG_VERSION",
]
