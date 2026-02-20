"""Foundation vision backbones (general-purpose) powered by Hugging Face."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel
except Exception as exc:  # pragma: no cover - optional dependency guard
    AutoConfig = AutoImageProcessor = AutoModel = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - executed when transformers is present
    _TRANSFORMERS_IMPORT_ERROR = None

from .head_builder import build_classification_head

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _ensure_transformers() -> None:
    if AutoModel is None or AutoConfig is None:
        raise ImportError(
            "transformers is required for foundation models. Install extras with `pip install -e .[advanced_dl]`."
        ) from _TRANSFORMERS_IMPORT_ERROR


@dataclass(frozen=True)
class FoundationGeneralSpec:
    key: str
    pretrained_id: str
    pooling: str
    input_size: tuple[int, int] = (224, 224)
    image_mean: Optional[tuple[float, float, float]] = None
    image_std: Optional[tuple[float, float, float]] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    family: str = "auto"
    auto_processor: bool = True
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    processor_call_kwargs: Dict[str, Any] = field(default_factory=dict)


class FoundationGeneralModel(nn.Module):
    """Wrapper around Hugging Face vision encoders with project-specific helpers."""

    def __init__(
        self,
        spec: FoundationGeneralSpec,
        *,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 0,
        classifier_dropout: float = 0.0,
        pool_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ensure_transformers()

        extra_processor_kwargs = kwargs.pop("processor_kwargs", None)
        extra_processor_call_kwargs = kwargs.pop("processor_call_kwargs", None)

        self.spec = spec
        self.num_classes = num_classes
        self.freeze_backbone_epochs = max(0, freeze_backbone_epochs)
        self.pool_type = (pool_type or spec.pooling).lower()
        self.current_epoch: int = 0
        self.recommended_input_size = spec.input_size
        self.processor = None
        self.processor_kwargs: Dict[str, Any] = dict(spec.processor_kwargs)
        self.processor_call_kwargs: Dict[str, Any] = dict(spec.processor_call_kwargs)
        if extra_processor_kwargs:
            self.processor_kwargs.update(extra_processor_kwargs)
        if extra_processor_call_kwargs:
            self.processor_call_kwargs.update(extra_processor_call_kwargs)

        load_kwargs = dict(spec.extra_model_kwargs)
        if cache_dir:
            load_kwargs.setdefault("cache_dir", cache_dir)
        if spec.revision:
            load_kwargs.setdefault("revision", spec.revision)
        load_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)

        try:
            if not pretrained:
                config = AutoConfig.from_pretrained(spec.pretrained_id, **load_kwargs)
                encoder = AutoModel.from_config(config, trust_remote_code=spec.trust_remote_code)
            else:
                encoder = AutoModel.from_pretrained(spec.pretrained_id, **load_kwargs)
        except ImportError as exc:  # pragma: no cover - surface missing deps
            raise ImportError(
                f"Failed to load foundation model '{spec.key}'. Ensure required dependencies are installed."
            ) from exc
        except Exception as exc:  # pragma: no cover - propagation with context
            raise RuntimeError(f"Unable to load foundation model '{spec.key}': {exc}") from exc

        if hasattr(encoder, "vision_model") and encoder.vision_model is not None:
            # CLIP-like models expose a multimodal wrapper; use the vision tower
            self.encoder = encoder.vision_model
        else:
            self.encoder = encoder

        try:
            signature = inspect.signature(self.encoder.forward)
            self._encoder_forward_params: set[str] = {
                name
                for name, param in signature.parameters.items()
                if param.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
        except (ValueError, TypeError):
            self._encoder_forward_params = set()

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "embed_dim", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "projection_dim", None)
        if hidden_size is None:
            raise ValueError(
                f"Unable to infer hidden size for foundation model '{spec.key}'. Please specify via extra_model_kwargs."
            )

        head_cfg = kwargs.pop("head", None)
        self.classifier = build_classification_head(
            hidden_size,
            num_classes,
            head_cfg=head_cfg,
            default_dropout=classifier_dropout,
        )

        # Normalization buffers (ImageNet -> model space)
        mean, std = self._infer_model_stats(spec)
        if spec.auto_processor:
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    spec.pretrained_id,
                    trust_remote_code=spec.trust_remote_code,
                    revision=spec.revision,
                    **self.processor_kwargs,
                )
            except Exception:
                self.processor = None
        self.register_buffer("imagenet_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("imagenet_std", IMAGENET_STD.clone(), persistent=False)
        self.register_buffer("model_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("model_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _infer_model_stats(self, spec: FoundationGeneralSpec) -> tuple[float, float, float]:
        mean = spec.image_mean
        std = spec.image_std
        if (mean is None or std is None) and spec.auto_processor:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    spec.pretrained_id,
                    trust_remote_code=spec.trust_remote_code,
                    revision=spec.revision,
                )
                if hasattr(processor, "image_mean") and processor.image_mean is not None:
                    mean = tuple(float(v) for v in processor.image_mean)
                if hasattr(processor, "image_std") and processor.image_std is not None:
                    std = tuple(float(v) for v in processor.image_std)
            except Exception:
                # Fall back to defaults defined in spec or ImageNet stats
                pass
        mean = mean or (0.485, 0.456, 0.406)
        std = std or (0.229, 0.224, 0.225)
        return mean, std

    def _denormalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Expected images tensor of shape (N, 3, H, W)")
        device = images.device
        return images * self.imagenet_std.to(device) + self.imagenet_mean.to(device)

    def _convert_inputs(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        x = self._denormalize_imagenet(images)
        x = (x - self.model_mean.to(device)) / self.model_std.to(device)
        return x

    def _pool_features(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pool = outputs.pooler_output
            if pool.ndim == 2:
                return pool
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Encoder output does not provide hidden states for pooling")
        if self.pool_type == "mean":
            return last_hidden.mean(dim=1)
        # default to CLS token
        return last_hidden[:, 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        images: torch.Tensor,
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> dict[str, torch.Tensor]:
        if self.spec.family == "clip_naflex":
            if self.processor is None:
                raise RuntimeError(
                    f"Model '{self.spec.key}' requires an image processor but none was initialized."
                )
            denorm = self._denormalize_imagenet(images)
            if original_sizes is None:
                sizes = [
                    (int(img.shape[-2]), int(img.shape[-1])) for img in denorm
                ]
            else:
                sizes = [(int(h), int(w)) for h, w in original_sizes]
                if len(sizes) != denorm.size(0):
                    raise ValueError(
                        "original_sizes length does not match batch size for NaFlex inputs."
                    )
            np_images = []
            for idx, img in enumerate(denorm):
                h, w = sizes[idx]
                if h <= 0 or w <= 0:
                    raise ValueError("original_sizes elements must be positive.")
                cropped = img[..., :h, :w]
                np_images.append(
                    (cropped.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                )
            call_kwargs = dict(self.processor_call_kwargs)
            call_kwargs.setdefault("return_tensors", "pt")
            processor_inputs = self.processor(images=np_images, **call_kwargs)
            aligned_inputs = self._align_processor_outputs(processor_inputs)
            return {k: v.to(images.device) for k, v in aligned_inputs.items()}

        pixel_values = self._convert_inputs(images)
        return {"pixel_values": pixel_values}

    def forward(
        self,
        images: torch.Tensor,
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(images, original_sizes=original_sizes)
        outputs = self.encoder(**inputs, return_dict=True)
        features = self._pool_features(outputs)
        logits = self.classifier(features)
        return logits


    def forward_features(
        self,
        images: torch.Tensor,
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(images, original_sizes=original_sizes)
        outputs = self.encoder(**inputs, return_dict=True)
        return self._pool_features(outputs)

    def _align_processor_outputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self._encoder_forward_params:
            return batch

        outputs: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            target_key = key

            if key == "pixel_attention_mask":
                if "pixel_attention_mask" in self._encoder_forward_params:
                    target_key = "pixel_attention_mask"
                elif "pixel_mask" in self._encoder_forward_params:
                    target_key = "pixel_mask"
                elif "attention_mask" in self._encoder_forward_params:
                    target_key = "attention_mask"
                else:
                    continue
            elif key == "pixel_mask":
                if "pixel_mask" in self._encoder_forward_params:
                    target_key = "pixel_mask"
                elif "pixel_attention_mask" in self._encoder_forward_params:
                    target_key = "pixel_attention_mask"
                elif "attention_mask" in self._encoder_forward_params:
                    target_key = "attention_mask"
                else:
                    continue
            elif key == "spatial_shapes" and "spatial_shapes" not in self._encoder_forward_params:
                continue
            elif key != "pixel_values" and key not in self._encoder_forward_params:
                continue

            outputs[target_key] = value

        return outputs



    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameter_groups(
        self,
        base_lr: float,
        *,
        head_lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        head_lr = head_lr if head_lr is not None else base_lr
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        groups: list[dict[str, Any]] = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": base_lr, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups or [{"params": self.parameters(), "lr": head_lr, "weight_decay": weight_decay}]

    def step_epoch(self) -> None:
        self.current_epoch += 1
        if self.freeze_backbone_epochs and self.current_epoch >= self.freeze_backbone_epochs:
            for param in self.encoder.parameters():
                if not param.requires_grad:
                    param.requires_grad = True


class FoundationGeneralRegistry:
    """Registry of supported general-purpose foundation backbones."""

    _registry: Dict[str, FoundationGeneralSpec] = {
        "siglip_base_patch16_384": FoundationGeneralSpec(
            key="siglip_base_patch16_384",
            pretrained_id="google/siglip-base-patch16-384",
            pooling="cls",
            input_size=(384, 384),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            family="clip",
        ),
        "siglip2_base_patch16_384": FoundationGeneralSpec(
            key="siglip2_base_patch16_384",
            pretrained_id="google/siglip2-base-patch16-384",
            pooling="cls",
            input_size=(384, 384),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            family="clip",
        ),
        "siglip2_so400m_patch14_384": FoundationGeneralSpec(
            key="siglip2_so400m_patch14_384",
            pretrained_id="google/siglip2-so400m-patch14-384",
            pooling="cls",
            input_size=(384, 384),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            family="clip",
        ),
        "siglip2_so400m_patch16_naflex": FoundationGeneralSpec(
            key="siglip2_so400m_patch16_naflex",
            pretrained_id="google/siglip2-so400m-patch16-naflex",
            pooling="cls",
            input_size=(512, 512),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            family="clip_naflex",
            processor_call_kwargs={"max_num_patches": 1024},
        ),
        "siglip2_giant_opt_patch16_384": FoundationGeneralSpec(
            key="siglip2_giant_opt_patch16_384",
            pretrained_id="google/siglip2-giant-opt-patch16-384",
            pooling="cls",
            input_size=(384, 384),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            family="clip",
        ),
        "mambavision_t2_1k": FoundationGeneralSpec(
            key="mambavision_t2_1k",
            pretrained_id="nvidia/MambaVision-T2-1K",
            pooling="cls",
            input_size=(448, 448),
            trust_remote_code=True,
        ),
        "dinov2_vit_l14": FoundationGeneralSpec(
            key="dinov2_vit_l14",
            pretrained_id="facebook/dinov2-large-imagenet1k-1-layer",
            pooling="cls",
            input_size=(336, 336),
        ),
    }

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> FoundationGeneralModel:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown foundation_general model '{key}'. Available: {available}")
        return FoundationGeneralModel(spec, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get_spec(cls, key: str) -> FoundationGeneralSpec:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown foundation_general model '{key}'. Available: {available}")
        return spec


__all__ = [
    "FoundationGeneralModel",
    "FoundationGeneralRegistry",
    "FoundationGeneralSpec",
]
