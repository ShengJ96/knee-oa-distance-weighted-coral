"""PyTorch Dataset and DataLoader for knee osteoarthritis classification.

This module provides PyTorch-compatible dataset classes for loading and
preprocessing knee X-ray images. Supports data augmentation, multiple
device targets (CPU/CUDA/MPS), and efficient batching.
"""

from __future__ import annotations

import inspect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# Optional MONAI-based train-time augmentation
try:
    from monai.transforms import RandAffine, RandAdjustContrast, RandGaussianNoise

    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False

# Optional Albumentations advanced augmentation
try:
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    _HAS_ALBUMENTATIONS = True
except Exception:
    A = None  # type: ignore
    ToTensorV2 = None  # type: ignore
    _HAS_ALBUMENTATIONS = False

import numpy as np
from .preprocessing import medical_preprocess_factory


def _ensure_resizable_tensor(value: Any) -> torch.Tensor:
    """Return a float32 tensor backed by resizable, contiguous storage."""

    if not torch.is_tensor(value):
        array = np.asarray(value)
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim == 3 and array.shape[0] not in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=True)
        tensor = torch.from_numpy(array)
    else:
        tensor = value.detach()

    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    try:
        storage = tensor.untyped_storage()
        if hasattr(storage, "resizable") and not storage.resizable():
            tensor = tensor.clone()
    except Exception:
        tensor = tensor.clone()

    return tensor


@dataclass
class PaddedBatch:
    """Batch container for variable-sized images padded to a common shape."""

    images: torch.Tensor
    labels: torch.Tensor
    sizes: torch.Tensor  # (N, 2) original (height, width)


def pad_image_collate(
    batch: List[Tuple[torch.Tensor, int]]
) -> PaddedBatch:
    """Collate function that pads variable-sized image tensors to a common size."""

    if not batch:
        raise ValueError("pad_image_collate received an empty batch.")

    images, labels = zip(*batch)
    channels = images[0].shape[0]
    max_height = max(img.shape[-2] for img in images)
    max_width = max(img.shape[-1] for img in images)

    padded = images[0].new_zeros((len(images), channels, max_height, max_width))
    sizes = torch.zeros(len(images), 2, dtype=torch.long)
    for idx, img in enumerate(images):
        _, h, w = img.shape
        padded[idx, :, :h, :w].copy_(img)
        sizes[idx, 0] = h
        sizes[idx, 1] = w

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return PaddedBatch(padded, labels_tensor, sizes)


def pad_image_collate(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate variable-sized image tensors by zero-padding to the max height/width."""

    if not batch:
        raise ValueError("pad_image_collate received an empty batch.")

    images, labels = zip(*batch)
    channels = images[0].shape[0]
    max_height = max(img.shape[-2] for img in images)
    max_width = max(img.shape[-1] for img in images)

    batch_tensor = images[0].new_zeros((len(images), channels, max_height, max_width))
    for idx, img in enumerate(images):
        _, h, w = img.shape
        batch_tensor[idx, :, :h, :w].copy_(img)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_tensor, labels_tensor


class KneeOADataset(Dataset):
    """PyTorch Dataset for knee osteoarthritis X-ray images.

    Expects directory structure: ``data_root/{split}/{grade}/*.png``. When using
    the default project layout, ``data_root`` should be a dataset-specific folder
    such as ``dataset/set_a``.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        limit_per_class: Optional[int] = None,
        seed: int = 42,
        pre_resize: bool = True,
    ):
        """Initialize dataset.

        Args:
            data_root: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Optional torchvision transforms
            target_size: Optional image resize target (height, width). When None, keep
                native resolution.
            limit_per_class: Limit samples per KL grade
            seed: Random seed for reproducibility
            pre_resize: Whether to resize images before applying transforms. Disable
                when the transform handles resizing internally (e.g., Albumentations).
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self.pre_resize = pre_resize

        # Set random seed
        random.seed(seed)

        # Class information
        self.num_classes = 5  # KL grades 0-4
        self.class_names = ["Normal", "Doubtful", "Minimal", "Moderate", "Severe"]

        # Load image paths and labels
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        self._load_data(limit_per_class)

    def _load_data(self, limit_per_class: Optional[int] = None) -> None:
        """Load image paths and labels from directory structure."""
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for grade in range(self.num_classes):
            grade_dir = split_dir / str(grade)
            if not grade_dir.exists():
                continue

            # Find all image files
            image_files = []
            for ext in [".png", ".jpg", ".jpeg"]:
                image_files.extend(grade_dir.glob(f"*{ext}"))

            # Shuffle and limit if specified
            random.shuffle(image_files)
            if limit_per_class is not None:
                image_files = image_files[:limit_per_class]

            # Add to dataset
            self.image_paths.extend(image_files)
            self.labels.extend([grade] * len(image_files))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            (image_tensor, label) tuple
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # Optional medical preprocessing (configurable via transform pipeline)
        if hasattr(self, "_medical_preprocess") and self._medical_preprocess:
            try:
                image = self._medical_preprocess(image)
            except Exception:
                pass

        # Resize image
        if self.pre_resize and self.target_size is not None:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image = transform(image)

        # Ensure output is a float tensor
        if isinstance(image, np.ndarray):
            array = image.astype(np.float32, copy=False)
            if array.ndim == 2:
                array = array[None, :, :]
            elif array.ndim == 3 and array.shape[0] not in (1, 3):
                array = np.transpose(array, (2, 0, 1))
            image = torch.from_numpy(array)
        elif isinstance(image, Image.Image):  # safeguard for rare custom transforms
            image = transforms.ToTensor()(image)

        if not torch.is_tensor(image):
            raise TypeError(
                f"Transform must return a torch.Tensor, got {type(image)!r}"
            )

        image = image.float().clone()

        label = self.labels[idx]
        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of samples per class."""
        from collections import Counter

        return dict(Counter(self.labels))

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced training."""
        class_counts = self.get_class_distribution()
        total_samples = len(self.labels)

        # Calculate inverse frequency weights
        weights = []
        for label in self.labels:
            weight = total_samples / (self.num_classes * class_counts[label])
            weights.append(weight)

        return torch.FloatTensor(weights)


class AlbumentationsAdapter:
    """Adapter to make Albumentations pipelines behave like torchvision transforms."""

    def __init__(self, pipeline):
        if not _HAS_ALBUMENTATIONS:
            raise ImportError(
                "Albumentations is not installed. Install the Stage 4 extras or set "
                "train_augmentation_library='torchvision'."
            )
        self.pipeline = pipeline
        # Signal to the dataset that the pipeline handles resizing internally
        self.skip_pre_resize = True

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"))
        augmented = self.pipeline(image=arr)
        result = augmented.get("image", augmented)

        if isinstance(result, torch.Tensor):
            tensor = result
        else:
            data = np.asarray(result)
            if data.ndim == 2:
                data = data[None, :, :]
            elif data.ndim == 3 and data.shape[0] not in (1, 3):
                data = np.transpose(data, (2, 0, 1))
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            if data.max() > 1.5:
                data /= 255.0
            tensor = torch.from_numpy(data)

        return _ensure_resizable_tensor(tensor)


def _random_resized_crop(height: int, width: int, **kwargs):
    """Construct RandomResizedCrop compatible with Albumentations v1/v2."""
    params = {}
    try:
        params = inspect.signature(A.RandomResizedCrop.__init__).parameters  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    if "size" in params:
        return A.RandomResizedCrop(size=(height, width), **kwargs)
    return A.RandomResizedCrop(height=height, width=width, **kwargs)


def _normalize_limit(limit: Any, *, base: float, additive: bool) -> Tuple[float, float]:
    """Convert scalar/tuple limits into absolute ranges."""
    if isinstance(limit, (int, float)):
        width = float(limit)
        if additive:
            return (base - width, base + width)
        return (-width, width)
    if isinstance(limit, (list, tuple)) and len(limit) == 2:
        low, high = float(limit[0]), float(limit[1])
        if additive:
            return (base + low, base + high)
        return (low, high)
    raise ValueError(f"Unsupported limit specification: {limit!r}")


def _shift_scale_rotate(**kwargs):
    """Approximate legacy ShiftScaleRotate with Affine to avoid deprecation warnings."""
    affine_cls = getattr(A, "Affine", None)
    if affine_cls is None:
        transform_cls = getattr(A, "ShiftScaleRotate", None)
        if transform_cls is None:
            raise AttributeError(
                "Albumentations is missing both Affine and ShiftScaleRotate transforms"
            )
        return transform_cls(**kwargs)

    shift_limit = kwargs.pop("shift_limit", 0.0)
    scale_limit = kwargs.pop("scale_limit", 0.0)
    rotate_limit = kwargs.pop("rotate_limit", 0.0)
    border_mode = kwargs.pop("border_mode", 0)
    value = kwargs.pop("value", 0)
    p = kwargs.get("p", 0.5)

    translate_range = _normalize_limit(shift_limit, base=0.0, additive=False)
    scale_range = _normalize_limit(scale_limit, base=1.0, additive=True)
    rotate_range = _normalize_limit(rotate_limit, base=0.0, additive=False)

    # Albumentations Affine accepts either tuple or float for rotate; provide tuple for variability
    sig = inspect.signature(affine_cls.__init__)
    params = sig.parameters

    affine_kwargs: Dict[str, Any] = {
        "scale": scale_range,
        "translate_percent": translate_range,
        "rotate": rotate_range,
        "fit_output": False,
        "p": p,
    }

    if "cval" in params:
        affine_kwargs["cval"] = value
    elif "fill" in params:
        affine_kwargs["fill"] = value

    if "mode" in params and "border_mode" not in params:
        mode_value: Any = border_mode
        if not isinstance(border_mode, str):
            cv2_to_str = {
                0: "constant",
                1: "reflect101",
                2: "reflect",
                3: "wrap",
                4: "replicate",
            }
            mode_value = cv2_to_str.get(border_mode, "constant")
        affine_kwargs["mode"] = mode_value
    elif "border_mode" in params:
        border_value: Any = border_mode
        if isinstance(border_mode, str):
            str_to_cv2 = {
                "constant": 0,
                "replicate": 4,
                "reflect": 2,
                "wrap": 3,
                "reflect101": 1,
            }
            border_value = str_to_cv2.get(border_mode.lower(), 0)
        affine_kwargs["border_mode"] = border_value

    if "shear" in kwargs and "shear" in params:
        affine_kwargs["shear"] = kwargs["shear"]

    if "mask_value" in kwargs:
        if "mask_value" in params:
            affine_kwargs["mask_value"] = kwargs["mask_value"]
        elif "fill_mask" in params:
            affine_kwargs["fill_mask"] = kwargs["mask_value"]

    return affine_cls(**affine_kwargs)


def _gaussian_noise(**kwargs):
    """Instantiate Gaussian/Gauss noise transform across versions."""
    transform_cls = getattr(A, "GaussianNoise", None)
    if transform_cls is None:
        transform_cls = getattr(A, "GaussNoise", None)
    if transform_cls is None:
        raise AttributeError(
            "Albumentations is missing Gaussian/Gauss noise transforms"
        )
    params = {}
    try:
        params = inspect.signature(transform_cls.__init__).parameters
    except (AttributeError, ValueError):
        pass

    converted = dict(kwargs)

    var_limit = converted.pop("var_limit", None)
    if var_limit is not None:
        if not isinstance(var_limit, (list, tuple)):
            var_limit = (0.0, float(var_limit))
        if "var_limit" in params:
            converted["var_limit"] = var_limit
        else:
            # Convert variance (in pixel scale) to std fraction in [0,1]
            std_min = math.sqrt(max(var_limit[0], 0.0))
            std_max = math.sqrt(max(var_limit[1], 0.0))
            std_range = (std_min / 255.0, std_max / 255.0)
            if "std_range" in params:
                converted["std_range"] = std_range
            elif "std_limit" in params:
                converted["std_limit"] = std_range

    mean = converted.pop("mean", None)
    if mean is not None:
        if "mean" in params:
            converted["mean"] = mean
        elif "mean_range" in params:
            mean_f = mean / 255.0
            converted["mean_range"] = (mean_f, mean_f)

    return transform_cls(**converted)


def _coarse_dropout(
    *,
    max_holes: int,
    max_height: int,
    max_width: int,
    min_holes: int = 1,
    min_height: Optional[int] = None,
    min_width: Optional[int] = None,
    fill_value: int = 0,
    p: float = 0.5,
):
    """Instantiate CoarseDropout compatible with legacy/new APIs."""
    transform_cls = getattr(A, "CoarseDropout", None)
    if transform_cls is None:
        raise AttributeError("Albumentations is missing CoarseDropout transform")

    params = {}
    try:
        params = inspect.signature(transform_cls.__init__).parameters
    except (AttributeError, ValueError):
        pass

    param_names = set(params.keys()) if params else set()

    if {"max_holes", "max_height", "max_width"}.issubset(
        param_names
    ) or not param_names:
        return transform_cls(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            min_holes=min_holes,
            fill_value=fill_value,
            p=p,
        )

    kwargs = {"p": p}

    if min_height is None:
        min_height = max(1, max_height // 2)
    if min_width is None:
        min_width = max(1, max_width // 2)

    if "num_holes_range" in param_names:
        kwargs["num_holes_range"] = (min_holes, max_holes)
    if "holes_number_range" in param_names:
        kwargs["holes_number_range"] = (min_holes, max_holes)
    if "hole_height_range" in param_names:
        kwargs["hole_height_range"] = (min_height, max_height)
    if "hole_width_range" in param_names:
        kwargs["hole_width_range"] = (min_width, max_width)
    if "min_holes" in param_names:
        kwargs["min_holes"] = min_holes
    if "fill" in param_names:
        kwargs["fill"] = fill_value
    if "fill_mask" in param_names:
        kwargs["fill_mask"] = fill_value
    if "max_holes" in param_names:
        kwargs["max_holes"] = max_holes
    if "max_num_holes" in param_names:
        kwargs["max_num_holes"] = max_holes
    if "max_height" in param_names:
        kwargs["max_height"] = max_height
    if "max_width" in param_names:
        kwargs["max_width"] = max_width
    if "min_height" in param_names:
        kwargs["min_height"] = min_height
    if "min_width" in param_names:
        kwargs["min_width"] = min_width

    # As a fallback, enforce relative ranges when absolute keys unavailable
    if "hole_height_range" not in kwargs and "hole_height" in param_names:
        kwargs["hole_height"] = max_height
    if "hole_width_range" not in kwargs and "hole_width" in param_names:
        kwargs["hole_width"] = max_width

    return transform_cls(**kwargs)


def _build_albumentations_transform(
    split: str,
    target_size: Optional[Tuple[int, int]],
    augment: bool,
    variant: str = "advanced",
) -> Any:
    if not _HAS_ALBUMENTATIONS or A is None or ToTensorV2 is None:
        raise ImportError(
            "Albumentations advanced transforms require the 'albumentations' "
            "and 'albumentations.pytorch' packages. Install the Stage 4 extras."
        )

    if target_size is None:
        raise ValueError(
            "Albumentations pipelines require an explicit target_size. "
            "Set data.target_size in the config or switch to torchvision transforms."
        )

    height, width = target_size
    ops: List[Any] = []

    if split == "train" and augment:
        key = variant.lower()
        if key == "light":
            ops.extend(
                [
                    _random_resized_crop(
                        height, width, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0
                    ),
                    A.HorizontalFlip(p=0.5),
                    _shift_scale_rotate(
                        shift_limit=0.03,
                        scale_limit=0.05,
                        rotate_limit=8,
                        border_mode=0,
                        value=0,
                        p=0.4,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=0.3
                    ),
                ]
            )
        else:  # advanced / strong
            ops.extend(
                [
                    _random_resized_crop(
                        height, width, scale=(0.8, 1.0), ratio=(0.9, 1.15), p=1.0
                    ),
                    A.HorizontalFlip(p=0.5),
                    _shift_scale_rotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=10,
                        border_mode=0,
                        value=0,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                            A.RandomBrightnessContrast(
                                brightness_limit=0.25, contrast_limit=0.25, p=1.0
                            ),
                            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                        ],
                        p=0.6,
                    ),
                    _gaussian_noise(var_limit=(0.0, 9.0), mean=0.0, p=0.3),
                    _coarse_dropout(
                        max_holes=4,
                        max_height=int(0.08 * height),
                        max_width=int(0.08 * width),
                        min_holes=1,
                        fill_value=0,
                        p=0.2,
                    ),
                ]
            )
    else:
        ops.append(A.Resize(height=height, width=width))

    ops.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(ops)


def get_data_transforms(
    split: str = "train",
    target_size: Optional[Tuple[int, int]] = (224, 224),
    augment: bool = True,
    medical_variant: str = "none",
    use_monai_train_aug: bool = False,
    augmentation_library: str = "torchvision",
    albumentations_variant: str = "advanced",
) -> transforms.Compose:
    """Get data preprocessing transforms.

    Args:
        split: Dataset split ('train', 'val', 'test')
        target_size: Target image size
        augment: Apply data augmentation for training
        medical_variant: Optional medical preprocessing strategy (reserved for future use)
        use_monai_train_aug: Enable legacy MONAI augmentation path for training
        augmentation_library: Augmentation backend ('torchvision', 'monai', 'albumentations')
        albumentations_variant: Strength preset when using albumentations ('light', 'advanced')

    Returns:
        Transform callable producing a normalized tensor
    """
    library = augmentation_library.lower()
    if split == "train" and use_monai_train_aug:
        library = "monai"

    if library not in {"torchvision", "monai", "albumentations"}:
        raise ValueError(
            f"Unsupported augmentation library '{augmentation_library}'. "
            "Choose from 'torchvision', 'monai', or 'albumentations'."
        )

    if library == "albumentations":
        pipeline = _build_albumentations_transform(
            split,
            target_size,
            augment if split == "train" else False,
            albumentations_variant,
        )
        return AlbumentationsAdapter(pipeline)

    if library == "monai":
        if not _HAS_MONAI:
            raise ImportError(
                "MONAI is not installed. Install with `pip install -e .[deep_learning]` "
                "or switch train_augmentation_library to 'torchvision'."
            )

        if split == "train" and augment:

            class MonaiAugmentAdapter:
                def __init__(self):
                    self.rotate_range = (-0.0873, 0.0873)  # ±5°
                    self.translate_range = (5.0, 5.0)
                    self.scale_range = (0.95, 1.05)
                    self.t2 = RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1))
                    self.t3 = RandGaussianNoise(prob=0.2, std=0.005)

                def __call__(self, pil_img: Image.Image):
                    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
                    arr = np.transpose(arr, (2, 0, 1))
                    h, w = arr.shape[-2], arr.shape[-1]
                    t1 = RandAffine(
                        prob=0.5,
                        rotate_range=self.rotate_range,
                        translate_range=self.translate_range,
                        scale_range=self.scale_range,
                        padding_mode="zeros",
                        mode="bilinear",
                        spatial_size=(h, w),
                    )
                    x = t1(arr)
                    x = self.t2(x)
                    x = self.t3(x)
                    import torch as _torch

                    if not isinstance(x, _torch.Tensor):
                        x = _torch.tensor(x, dtype=_torch.float32)
                    mean = _torch.tensor([0.485, 0.456, 0.406], dtype=_torch.float32)[
                        :, None, None
                    ]
                    std = _torch.tensor([0.229, 0.224, 0.225], dtype=_torch.float32)[
                        :, None, None
                    ]
                    x = (x - mean) / std
                    return x

            return transforms.Compose([MonaiAugmentAdapter()])

        # Fall back to torchvision normalization for val/test when MONAI is selected
        library = "torchvision"

    # Torchvision defaults
    resize_ops: List[Any] = []
    if target_size is not None:
        resize_ops.append(transforms.Resize(target_size))

    if split == "train" and augment:
        return transforms.Compose(
            resize_ops
            + [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transforms.Compose(
        resize_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def create_data_loaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    num_workers: int = 4,
    augment_train: bool = True,
    limit_per_class: Optional[int] = None,
    limit_per_class_val: Optional[int] = None,
    limit_per_class_test: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    medical_variant: str = "none",
    use_monai_train_aug: bool = False,
    train_augmentation_library: str = "torchvision",
    eval_augmentation_library: Optional[str] = None,
    albumentations_variant: str = "advanced",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    collate_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        data_root: Path to dataset directory
        batch_size: Batch size for training
        target_size: Target image size
        num_workers: Number of worker processes
        augment_train: Apply data augmentation to training set
        limit_per_class: Limit samples per class in train split (for debugging)
        limit_per_class_val: Limit samples per class in validation split
        limit_per_class_test: Limit samples per class in test split
        pin_memory: Use pinned memory for faster GPU transfer
        medical_variant: Optional medical preprocessing recipe
        use_monai_train_aug: Backwards-compatible flag to enable MONAI pipeline
        train_augmentation_library: One of {'torchvision', 'monai', 'albumentations'}
        eval_augmentation_library: Override evaluation library (defaults to training choice)
        albumentations_variant: Intensity of albumentations pipeline ('light', 'advanced')

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    # Determine pin_memory if not explicitly provided: only enable on CUDA
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())
    # Determine augmentation libraries
    train_library = (train_augmentation_library or "torchvision").lower()
    if use_monai_train_aug:
        train_library = "monai"

    eval_library = (
        eval_augmentation_library.lower()
        if eval_augmentation_library
        else ("albumentations" if train_library == "albumentations" else "torchvision")
    )

    # Create transforms
    train_transform = get_data_transforms(
        "train",
        target_size,
        augment_train,
        medical_variant,
        use_monai_train_aug,
        augmentation_library=train_library,
        albumentations_variant=albumentations_variant,
    )
    val_transform = get_data_transforms(
        "val",
        target_size,
        False,
        medical_variant,
        False,
        augmentation_library=eval_library,
        albumentations_variant=albumentations_variant,
    )
    test_transform = get_data_transforms(
        "test",
        target_size,
        False,
        medical_variant,
        False,
        augmentation_library=eval_library,
        albumentations_variant=albumentations_variant,
    )

    # Create datasets
    train_pre_resize = target_size is not None and train_library != "albumentations"
    eval_pre_resize = target_size is not None and eval_library != "albumentations"
    train_dataset = KneeOADataset(
        data_root,
        "train",
        train_transform,
        target_size,
        limit_per_class,
        pre_resize=train_pre_resize,
    )
    train_dataset._medical_preprocess = medical_preprocess_factory(medical_variant)
    val_dataset = KneeOADataset(
        data_root,
        "val",
        val_transform,
        target_size,
        limit_per_class=limit_per_class_val,
        pre_resize=eval_pre_resize,
    )
    val_dataset._medical_preprocess = medical_preprocess_factory(medical_variant)
    test_dataset = KneeOADataset(
        data_root,
        "test",
        test_transform,
        target_size,
        limit_per_class=limit_per_class_test,
        pre_resize=eval_pre_resize,
    )
    test_dataset._medical_preprocess = medical_preprocess_factory(medical_variant)

    use_distributed = distributed and world_size > 1

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    test_sampler: Optional[DistributedSampler] = None

    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Ensure consistent batch sizes
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def create_multi_source_data_loaders(
    datasets: List[Dict[str, Any]],
    *,
    batch_size: int = 32,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    num_workers: int = 4,
    augment_train: bool = True,
    pin_memory: Optional[bool] = None,
    default_medical_variant: str = "none",
    train_augmentation_library: str = "torchvision",
    eval_augmentation_library: Optional[str] = None,
    albumentations_variant: str = "advanced",
    sampling_strategy: str = "proportional",
    collate_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders that merge multiple dataset roots.

    Args:
        datasets: List of dataset specifications. Each entry supports:
            - ``root`` (required): data root directory.
            - ``weight`` (optional, float): sampling weight for training.
            - ``medical_variant`` (optional): overrides default preprocessing.
            - ``limit_per_class``/``limit_per_class_val``/``limit_per_class_test`` overrides.
        sampling_strategy: ``"proportional"`` keeps the natural class counts
            across datasets. ``"balanced"`` balances per-dataset contribution
            via ``WeightedRandomSampler`` using dataset-level weights.
    """

    if not datasets:
        raise ValueError("`datasets` must contain at least one dataset spec")

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    train_library = (train_augmentation_library or "torchvision").lower()
    eval_library = (
        eval_augmentation_library.lower()
        if eval_augmentation_library
        else ("albumentations" if train_library == "albumentations" else "torchvision")
    )

    train_transform = get_data_transforms(
        "train",
        target_size,
        augment_train,
        default_medical_variant,
        train_library == "monai",
        augmentation_library=train_library,
        albumentations_variant=albumentations_variant,
    )
    val_transform = get_data_transforms(
        "val",
        target_size,
        False,
        default_medical_variant,
        False,
        augmentation_library=eval_library,
        albumentations_variant=albumentations_variant,
    )
    test_transform = get_data_transforms(
        "test",
        target_size,
        False,
        default_medical_variant,
        False,
        augmentation_library=eval_library,
        albumentations_variant=albumentations_variant,
    )

    train_pre_resize = target_size is not None and train_library != "albumentations"
    eval_pre_resize = target_size is not None and eval_library != "albumentations"

    train_datasets: List[KneeOADataset] = []
    val_datasets: List[KneeOADataset] = []
    test_datasets: List[KneeOADataset] = []
    dataset_weights: List[float] = []

    for spec in datasets:
        root = spec.get("root")
        if not root:
            raise ValueError("Each multi-source dataset spec must include 'root'")
        medical_variant = spec.get("medical_variant", default_medical_variant)
        weight = float(spec.get("weight", 1.0))
        dataset_weights.append(max(weight, 0.0))

        limit_train = spec.get("limit_per_class")
        limit_val = spec.get("limit_per_class_val")
        limit_test = spec.get("limit_per_class_test")

        train_ds = KneeOADataset(
            root,
            "train",
            train_transform,
            target_size,
            limit_per_class=limit_train,
            pre_resize=train_pre_resize,
        )
        train_ds._medical_preprocess = medical_preprocess_factory(medical_variant)
        train_datasets.append(train_ds)

        val_ds = KneeOADataset(
            root,
            "val",
            val_transform,
            target_size,
            limit_per_class=limit_val,
            pre_resize=eval_pre_resize,
        )
        val_ds._medical_preprocess = medical_preprocess_factory(medical_variant)
        val_datasets.append(val_ds)

        test_ds = KneeOADataset(
            root,
            "test",
            test_transform,
            target_size,
            limit_per_class=limit_test,
            pre_resize=eval_pre_resize,
        )
        test_ds._medical_preprocess = medical_preprocess_factory(medical_variant)
        test_datasets.append(test_ds)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_sampler = None
    shuffle = True
    if sampling_strategy.lower() == "balanced" and len(train_datasets) > 1:
        lengths = [len(ds) for ds in train_datasets]
        if any(length == 0 for length in lengths):
            raise ValueError(
                "Balanced sampling requires each dataset to contain samples"
            )
        total_weight = sum(dataset_weights) or float(len(train_datasets))
        scaled_weights = [w if w > 0 else 1.0 for w in dataset_weights]
        if total_weight <= 0:
            total_weight = float(len(train_datasets))
        weights: List[float] = []
        for length, dataset_weight in zip(lengths, scaled_weights):
            base = dataset_weight / length
            weights.extend([base] * length)
        weight_tensor = torch.tensor(weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=len(weight_tensor),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device for training.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU

    Returns:
        torch.device object
    """
    if not prefer_gpu:
        return torch.device("cpu")

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (Apple Silicon GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback to CPU
    return torch.device("cpu")


def print_dataset_info(data_root: Union[str, Path]) -> None:
    """Print dataset information and statistics.

    Args:
        data_root: Path to dataset directory
    """
    print("=" * 60)
    print("📊 Dataset Information")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        try:
            dataset = KneeOADataset(data_root, split)
            class_dist = dataset.get_class_distribution()

            print(f"\n{split.upper()} SET:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Class distribution:")
            for grade, count in class_dist.items():
                class_name = dataset.class_names[grade]
                percentage = count / len(dataset) * 100
                print(f"    Grade {grade} ({class_name}): {count} ({percentage:.1f}%)")

        except FileNotFoundError:
            print(f"\n{split.upper()} SET: ❌ Not found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage and testing
    import argparse

    parser = argparse.ArgumentParser(description="Test PyTorch dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/set_a",
        help="Path to dataset directory (e.g. dataset/set_a)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for testing"
    )
    args = parser.parse_args()

    # Print dataset info
    print_dataset_info(args.data_root)

    # Test data loading
    print("\n🔄 Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        # Test one batch
        for images, labels in train_loader:
            print(f"✅ Batch loaded successfully:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Device: {get_device()}")
            break

    except Exception as e:
        print(f"❌ Error loading data: {e}")
