"""Advanced convolutional architectures powered by timm.

Stage 4 introduces stronger backbones (EfficientNet, ConvNeXt, RegNet)
with flexible freezing strategies and metadata that keeps experiment
configuration consistent with notebooks and CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

try:  # Optional dependency resolved in Stage 4 install extras
    import timm  # type: ignore
except Exception as exc:  # pragma: no cover - import guarded for optional dep
    timm = None
    _TIMM_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - executed only when timm is present
    _TIMM_IMPORT_ERROR = None

DEFAULT_TRAINABLE_PATTERNS = ["head", "fc", "classifier"]


@dataclass(frozen=True)
class AdvancedCNNSpec:
    """Specification for a timm backbone used in Stage 4."""

    key: str
    timm_name: str
    input_size: int
    description: str
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainable_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_TRAINABLE_PATTERNS))


class AdvancedCNNModel(nn.Module):
    """Wrapper around timm models with OA-specific helpers."""

    def __init__(
        self,
        timm_name: str,
        *,
        num_classes: int = 5,
        pretrained: bool = True,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        global_pool: str = "avg",
        classifier_dropout: Optional[float] = None,
        freeze_backbone: bool = False,
        trainable_patterns: Optional[Iterable[str]] = None,
        checkpoint_path: Optional[str] = None,
        create_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for AdvancedCNNModel. Install the optional "
                "dependency set with `pip install -e .[advanced_dl]`."
            ) from _TIMM_IMPORT_ERROR

        timm_kwargs: Dict[str, Any] = {
            "num_classes": num_classes,
            "in_chans": in_chans,
            "pretrained": pretrained,
            "drop_rate": drop_rate,
            "drop_path_rate": drop_path_rate,
            "global_pool": global_pool,
        }
        if create_kwargs:
            timm_kwargs.update(create_kwargs)

        effective_num_classes = int(timm_kwargs.get("num_classes", num_classes))

        self.model = timm.create_model(timm_name, **timm_kwargs)
        self.model_name = timm_name
        self.num_classes = effective_num_classes
        self.trainable_patterns = [s.lower() for s in (trainable_patterns or DEFAULT_TRAINABLE_PATTERNS)]

        if classifier_dropout is not None:
            self._inject_classifier_dropout(classifier_dropout)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)  # type: ignore[attr-defined]
        raise NotImplementedError(f"forward_features is not exposed by {self.model_name}")

    def get_classifier(self) -> nn.Module:
        classifier = getattr(self.model, "get_classifier", None)
        if callable(classifier):
            cls = classifier()
            if isinstance(cls, nn.Module):
                return cls
        for attr in ("classifier", "head", "fc"):
            module = getattr(self.model, attr, None)
            if isinstance(module, nn.Module):
                if attr == "head" and hasattr(module, "fc"):
                    sub = getattr(module, "fc")
                    if isinstance(sub, nn.Module):
                        return sub
                return module
        raise AttributeError(f"Unable to locate classifier head for {self.model_name}")

    def freeze_backbone(self, *, trainable_patterns: Optional[Iterable[str]] = None) -> None:
        patterns = [s.lower() for s in (trainable_patterns or self.trainable_patterns)]
        classifier_ids = {id(p) for p in self.get_classifier().parameters() if p.requires_grad}
        for name, param in self.model.named_parameters():
            if id(param) in classifier_ids:
                param.requires_grad = True
                continue
            lowered = name.lower()
            if any(pattern in lowered for pattern in patterns):
                param.requires_grad = True
            else:
                param.requires_grad = False

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
        head_lr = base_lr if head_lr is None else head_lr
        classifier_params = list(self.get_classifier().parameters())
        classifier_ids = {id(p) for p in classifier_params}

        backbone_params = [p for p in self.model.parameters() if id(p) not in classifier_ids and p.requires_grad]
        head_params = [p for p in classifier_params if p.requires_grad]

        groups: List[Dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups if groups else [{"params": classifier_params, "lr": head_lr, "weight_decay": weight_decay}]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inject_classifier_dropout(self, dropout: float) -> None:
        classifier = self.get_classifier()
        if isinstance(classifier, nn.Linear):
            new_head = nn.Sequential(nn.Dropout(dropout), classifier)
            self._replace_classifier(new_head)
        elif isinstance(classifier, nn.Sequential):
            classifier.insert(0, nn.Dropout(dropout))
        else:
            self._replace_classifier(nn.Sequential(nn.Dropout(dropout), classifier))

    def _replace_classifier(self, new_module: nn.Module) -> None:
        if hasattr(self.model, "reset_classifier"):
            # Many timm models expose reset_classifier(num_classes, global_pool)
            in_features = getattr(self.model, "num_features", None)
            if isinstance(new_module, nn.Linear) and in_features is not None and new_module.in_features != in_features:
                raise ValueError("Replacement classifier has mismatched input features")
        for attr in ("classifier", "head", "fc"):
            module = getattr(self.model, attr, None)
            if isinstance(module, nn.Module):
                if attr == "head" and hasattr(module, "fc"):
                    setattr(module, "fc", new_module)
                else:
                    setattr(self.model, attr, new_module)
                return
        raise AttributeError(f"Unable to replace classifier for {self.model_name}")

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ Missing keys when loading {self.model_name} from {path}: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys when loading {self.model_name} from {path}: {unexpected}")


class AdvancedCNNRegistry:
    """Factory for Stage 4 advanced CNNs."""

    _registry: Dict[str, AdvancedCNNSpec] = {
        "efficientnet_b0": AdvancedCNNSpec(
            key="efficientnet_b0",
            timm_name="efficientnet_b0",
            input_size=224,
            description="EfficientNet-B0 baseline with lightweight compute.",
            default_kwargs={"drop_rate": 0.2, "drop_path_rate": 0.2},
        ),
        "efficientnet_b3": AdvancedCNNSpec(
            key="efficientnet_b3",
            timm_name="efficientnet_b3",
            input_size=300,
            description="EfficientNet-B3 higher-resolution variant for OA grading.",
            default_kwargs={"drop_rate": 0.3, "drop_path_rate": 0.3},
        ),
        "convnext_tiny": AdvancedCNNSpec(
            key="convnext_tiny",
            timm_name="convnext_tiny",
            input_size=224,
            description="ConvNeXt-Tiny with conv-attention hybrid blocks.",
            default_kwargs={"drop_path_rate": 0.4},
            trainable_patterns=["head", "classifier", "norm"],
        ),
        "convnext_small": AdvancedCNNSpec(
            key="convnext_small",
            timm_name="convnext_small",
            input_size=224,
            description="ConvNeXt-Small for stronger high-resolution modeling.",
            default_kwargs={"drop_path_rate": 0.5},
            trainable_patterns=["head", "classifier", "norm"],
        ),
        "regnety_008": AdvancedCNNSpec(
            key="regnety_008",
            timm_name="regnety_008",
            input_size=224,
            description="RegNetY-8.0GF for balanced accuracy/compute trade-offs.",
            default_kwargs={"drop_path_rate": 0.2},
        ),
    }

    @classmethod
    def create(
        cls,
        key: str,
        *,
        override_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> AdvancedCNNModel:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown advanced CNN '{key}'. Available: {available}")

        wrapper_keys = {"freeze_backbone", "classifier_dropout", "trainable_patterns", "checkpoint_path"}
        create_kwargs: Dict[str, Any] = dict(spec.default_kwargs)
        if override_kwargs:
            create_kwargs.update(override_kwargs)

        wrapper_kwargs: Dict[str, Any] = {}
        for key_name, value in model_kwargs.items():
            if key_name in wrapper_keys:
                wrapper_kwargs[key_name] = value
            else:
                create_kwargs[key_name] = value

        trainable_patterns = list(wrapper_kwargs.pop("trainable_patterns", spec.trainable_patterns))

        model = AdvancedCNNModel(
            spec.timm_name,
            trainable_patterns=trainable_patterns,
            create_kwargs=create_kwargs,
            **wrapper_kwargs,
        )
        # NB: AdvancedCNNModel already consumes relevant kwargs via additional_create_kwargs.
        # Attach metadata for downstream tooling (CLI/configs, logging)
        model.metadata = {
            "model_key": key,
            "timm_name": spec.timm_name,
            "recommended_input_size": spec.input_size,
            "description": spec.description,
        }
        return model

    @classmethod
    def get_spec(cls, key: str) -> AdvancedCNNSpec:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown advanced CNN '{key}'. Available: {available}")
        return spec

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def model_metadata(cls, key: str) -> Dict[str, Any]:
        spec = cls.get_spec(key)
        return {
            "model_key": spec.key,
            "timm_name": spec.timm_name,
            "input_size": spec.input_size,
            "description": spec.description,
        }


__all__ = ["AdvancedCNNModel", "AdvancedCNNRegistry", "AdvancedCNNSpec"]


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    if timm is None:
        raise SystemExit("Install Stage 4 extras to run the advanced CNN smoke test.")
    sample_model = AdvancedCNNRegistry.create("efficientnet_b0", num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = sample_model(dummy)
    print("Output shape:", out.shape)
