"""Vision transformer architectures for Stage 4 advanced experiments.

This module wraps Hugging Face `transformers` implementations (ViT, Swin)
with convenience utilities tailored for the knee osteoarthritis project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:  # Optional dependency introduced in Stage 4 advanced DL extras
    from transformers import (
        ViTConfig,
        ViTForImageClassification,
        SwinConfig,
        SwinForImageClassification,
    )
except Exception as exc:  # pragma: no cover - optional import guard
    ViTConfig = ViTForImageClassification = SwinConfig = SwinForImageClassification = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - executed when transformers is present
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass(frozen=True)
class VisionTransformerSpec:
    """Specification describing a supported transformer backbone."""

    key: str
    architecture: str  # 'vit' or 'swin'
    pretrained_id: Optional[str]
    image_size: int
    config_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_from_pretrained_kwargs: Dict[str, Any] = field(default_factory=dict)

    def ensure_transformers(self) -> None:
        if ViTForImageClassification is None or SwinForImageClassification is None:
            raise ImportError(
                "transformers is required for VisionTransformer models. "
                "Install Stage 4 dependencies with `pip install -e .[advanced_dl]`."
            ) from _TRANSFORMERS_IMPORT_ERROR


class VisionTransformerModel(nn.Module):
    """Wrapper around ViT/Swin classification models with project helpers."""

    def __init__(
        self,
        spec: VisionTransformerSpec,
        *,
        num_classes: int = 5,
        pretrained: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        ignore_mismatched_sizes: bool = True,
        output_hidden_states: bool = False,
        config_overrides: Optional[Dict[str, Any]] = None,
        from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        spec.ensure_transformers()

        self.spec = spec
        self.num_classes = num_classes
        self.output_hidden_states = output_hidden_states
        self.cache_dir = cache_dir
        self.ignore_mismatched_sizes = ignore_mismatched_sizes

        overrides = dict(config_overrides or {})
        overrides.setdefault("num_labels", num_classes)
        overrides.setdefault("image_size", spec.image_size)
        overrides.setdefault("label2id", {str(i): i for i in range(num_classes)})
        overrides.setdefault("id2label", {i: str(i) for i in range(num_classes)})

        if spec.architecture == "vit":
            config_cls = ViTConfig
            model_cls = ViTForImageClassification
        elif spec.architecture == "swin":
            config_cls = SwinConfig
            model_cls = SwinForImageClassification
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported architecture: {spec.architecture}")

        if pretrained:
            model_id = pretrained_checkpoint or spec.pretrained_id
            if not model_id:
                raise ValueError(
                    f"Pretrained weights requested for '{spec.key}' but no pretrained_id or checkpoint provided."
                )
            load_kwargs = dict(spec.default_from_pretrained_kwargs)
            load_kwargs.update(from_pretrained_kwargs or {})
            load_kwargs.setdefault("ignore_mismatched_sizes", ignore_mismatched_sizes)
            if cache_dir:
                load_kwargs.setdefault("cache_dir", cache_dir)
            load_kwargs.setdefault("num_labels", num_classes)
            load_kwargs.setdefault("label2id", overrides["label2id"])
            load_kwargs.setdefault("id2label", overrides["id2label"])
            self.model = model_cls.from_pretrained(model_id, **load_kwargs)
        else:
            config_kwargs = dict(spec.config_kwargs)
            config_kwargs.update(overrides)
            for key, value in overrides.items():
                config_kwargs.setdefault(key, value)
            config = config_cls(**config_kwargs)
            self.model = model_cls(config)

        # Ensure classifier head matches the target number of classes
        self.model.config.num_labels = num_classes
        classifier = self.get_classifier()
        if getattr(classifier, "out_features", None) != num_classes:
            in_features = getattr(classifier, "in_features", None)
            if in_features is None:
                raise AttributeError("Unable to determine classifier in_features for replacement")
            new_classifier = nn.Linear(in_features, num_classes)
            nn.init.trunc_normal_(new_classifier.weight, std=0.02)
            if new_classifier.bias is not None:
                nn.init.zeros_(new_classifier.bias)
            setattr(self.model, "classifier", new_classifier)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=self.output_hidden_states,
            return_dict=True,
        )
        self._last_hidden_states = outputs.hidden_states  # type: ignore[attr-defined]
        return outputs.logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states:
            # Return final hidden state (CLS token for ViT, pooled for Swin)
            last_state = hidden_states[-1]
            if self.spec.architecture == "vit":
                return last_state[:, 0]  # CLS token embedding
            return last_state.mean(dim=1)
        # Fallback to logits if hidden states are unavailable
        return outputs.logits

    def get_classifier(self) -> nn.Module:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, nn.Module):
            raise AttributeError("Transformer model does not expose a classifier module")
        return classifier

    def freeze_backbone(self) -> None:
        classifier_params = {id(p) for p in self.get_classifier().parameters()}
        for param in self.model.parameters():
            param.requires_grad = id(param) in classifier_params

    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def trainable_parameter_groups(
        self,
        base_lr: float,
        *,
        head_lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        head_lr = head_lr if head_lr is not None else base_lr
        classifier_params = list(self.get_classifier().parameters())
        classifier_ids = {id(p) for p in classifier_params}
        backbone_params = [p for p in self.model.parameters() if id(p) not in classifier_ids and p.requires_grad]
        groups: List[Dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups


class VisionTransformerRegistry:
    """Registry for supported Stage 4 transformer backbones."""

    _registry: Dict[str, VisionTransformerSpec] = {
        "vit_b16": VisionTransformerSpec(
            key="vit_b16",
            architecture="vit",
            pretrained_id="google/vit-base-patch16-224",
            image_size=224,
            config_kwargs={
                "patch_size": 16,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "qkv_bias": True,
            },
        ),
        "swin_t": VisionTransformerSpec(
            key="swin_t",
            architecture="swin",
            pretrained_id="microsoft/swin-tiny-patch4-window7-224",
            image_size=224,
            config_kwargs={
                "patch_size": 4,
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 7,
                "drop_path_rate": 0.2,
            },
        ),
    }

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def get_spec(cls, key: str) -> VisionTransformerSpec:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown vision transformer '{key}'. Available: {available}")
        return spec

    @classmethod
    def create(
        cls,
        key: str,
        **model_kwargs: Any,
    ) -> VisionTransformerModel:
        spec = cls.get_spec(key)
        return VisionTransformerModel(spec, **model_kwargs)

    @classmethod
    def model_metadata(cls, key: str) -> Dict[str, Any]:
        spec = cls.get_spec(key)
        return {
            "model_key": spec.key,
            "architecture": spec.architecture,
            "pretrained_id": spec.pretrained_id,
            "image_size": spec.image_size,
        }


__all__ = [
    "VisionTransformerModel",
    "VisionTransformerRegistry",
    "VisionTransformerSpec",
]


if __name__ == "__main__":  # pragma: no cover - quick smoke helper
    if ViTForImageClassification is None:
        raise SystemExit("Install transformers extras to run the vision transformer smoke test.")
    model = VisionTransformerRegistry.create("vit_b16", pretrained=False, num_classes=5)
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print("Logits shape:", logits.shape)
