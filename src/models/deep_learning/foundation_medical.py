"""Medical imaging foundation backbones (BiomedCLIP, BioViL, RadImageNet)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .foundation_general import (
    FoundationGeneralModel,
    FoundationGeneralSpec,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .head_builder import build_classification_head

try:
    import timm
except Exception:  # pragma: no cover - optional dependency guard
    timm = None

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency guard
    hf_hub_download = None  # type: ignore

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
except Exception:
    AutoConfig = AutoImageProcessor = AutoModel = AutoProcessor = None  # type: ignore

from .biovil_image import BioViLTImageModel

try:
    import transformers.models.clip.modeling_clip as _clip_mod  # type: ignore[attr-defined]
    from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings  # type: ignore[attr-defined]

    if hasattr(_clip_mod, "__all__") and "CLIPVisionEmbeddings" not in _clip_mod.__all__:
        _clip_mod.__all__ = list(_clip_mod.__all__) + ["CLIPVisionEmbeddings"]  # type: ignore[assignment]

    if not hasattr(_clip_mod, "CLIPOutput"):
        from transformers.utils import ModelOutput  # type: ignore[attr-defined]

        @dataclass
        class _CompatCLIPOutput(ModelOutput):
            loss: Optional[torch.Tensor] = None
            logits_per_image: Optional[torch.Tensor] = None
            logits_per_text: Optional[torch.Tensor] = None
            text_embeds: Optional[torch.Tensor] = None
            image_embeds: Optional[torch.Tensor] = None
            text_model_output: Optional[Any] = None
            vision_model_output: Optional[Any] = None

        _clip_mod.CLIPOutput = _CompatCLIPOutput  # type: ignore[attr-defined]
    required_exports = ["CLIPVisionEmbeddings", "CLIPOutput", "CLIPMLP"]
    if hasattr(_clip_mod, "__all__"):
        exported = set(_clip_mod.__all__)
        for name in required_exports:
            if name in exported:
                continue
            if hasattr(_clip_mod, name):
                _clip_mod.__all__ = list(_clip_mod.__all__) + [name]  # type: ignore[assignment]
                exported.add(name)
except Exception:  # pragma: no cover - optional dependency guard
    pass


@dataclass(frozen=True)
class FoundationMedicalSpec:
    key: str
    family: str
    pretrained_id: Optional[str] = None
    pooling: str = "cls"
    input_size: tuple[int, int] = (224, 224)
    image_mean: Optional[tuple[float, float, float]] = None
    image_std: Optional[tuple[float, float, float]] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)


def _remap_radimagenet_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert RadImageNet checkpoint keys to timm ResNet naming."""

    mapping = {
        "0": "conv1",
        "1": "bn1",
        "4": "layer1",
        "5": "layer2",
        "6": "layer3",
        "7": "layer4",
    }

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            parts = key.split(".", 2)
            if len(parts) < 3:
                continue
            block = parts[1]
            remainder = parts[2]
            new_prefix = mapping.get(block)
            if new_prefix is None:
                continue
            new_key = f"{new_prefix}.{remainder}"
            remapped[new_key] = value
        elif key.startswith("fc") or key.startswith("classifier"):
            # Drop classification head parameters; we re-initialize our own head.
            continue
        else:
            remapped[key] = value

    return remapped or state_dict


class FoundationMedicalModel(nn.Module):
    """Wrap medical foundation backbones with project-friendly classifier."""

    def __init__(
        self,
        spec: FoundationMedicalSpec,
        *,
        num_classes: int = 5,
        pretrained: bool = True,
        classifier_dropout: float = 0.0,
        freeze_backbone_epochs: int = 0,
        checkpoint_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.num_classes = num_classes
        self.freeze_backbone_epochs = max(0, freeze_backbone_epochs)
        self.pool_type = spec.pooling
        self.current_epoch: int = 0
        self.recommended_input_size = spec.input_size

        processor_kwargs: Dict[str, Any] = {}
        if cache_dir:
            processor_kwargs.setdefault("cache_dir", cache_dir)
        if spec.revision:
            processor_kwargs.setdefault("revision", spec.revision)
        processor_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)

        if spec.family == "clip":
            if AutoModel is None:
                raise ImportError(
                    "transformers is required for medical foundation models. Install extras with `pip install -e .[advanced_dl]`."
                )

            load_kwargs = dict(spec.extra_model_kwargs)
            load_kwargs.update(processor_kwargs)

            if not pretrained:
                config = AutoConfig.from_pretrained(spec.pretrained_id, **load_kwargs)
                backbone = AutoModel.from_config(config)
            else:
                backbone = AutoModel.from_pretrained(spec.pretrained_id, **load_kwargs)

            if hasattr(backbone, "vision_model") and backbone.vision_model is not None:
                self.encoder = backbone.vision_model
            else:
                self.encoder = backbone

            hidden_size = getattr(self.encoder.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.encoder.config, "embed_dim", None)
        elif spec.family == "biovil":
            if hf_hub_download is None:
                raise ImportError(
                    "huggingface_hub is required for BioViL-T backbones. Install with `pip install huggingface_hub`."
                )

            weights_filename = spec.extra_model_kwargs.get(
                "weights_filename", "biovil_t_image_model_proj_size_128.pt"
            )
            download_kwargs: Dict[str, Any] = {}
            if cache_dir:
                download_kwargs["cache_dir"] = cache_dir
            if spec.revision:
                download_kwargs["revision"] = spec.revision

            try:
                weights_path = hf_hub_download(
                    repo_id=spec.pretrained_id,
                    filename=weights_filename,
                    **download_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to download BioViL-T weights '{weights_filename}' from {spec.pretrained_id}: {exc}"
                ) from exc

            self.encoder = BioViLTImageModel(weights_path)
            hidden_size = getattr(self.encoder, "feature_dim", None)
        elif spec.family == "timm_resnet":
            if timm is None:
                raise ImportError(
                    "timm is required for RadImageNet backbones. Install extras with `pip install -e .[advanced_dl]`."
                )
            model_name = spec.extra_model_kwargs.get("model_name", "resnet50")
            weights_filename = spec.extra_model_kwargs.get("weights_filename")
            self.encoder = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool="avg"
            )

            state_dict = None
            if pretrained:
                ckpt_path = Path(checkpoint_file) if checkpoint_file else None
                if ckpt_path and ckpt_path.is_file():
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                elif spec.pretrained_id:
                    if hf_hub_download is None:
                        raise ImportError(
                            "huggingface_hub is required to download RadImageNet weights. Install with `pip install huggingface_hub`."
                        )
                    download_kwargs: Dict[str, Any] = {}
                    if cache_dir:
                        download_kwargs["cache_dir"] = cache_dir
                    filename = weights_filename or "ResNet50.pt"
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=spec.pretrained_id,
                            filename=filename,
                            **download_kwargs,
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"Failed to download RadImageNet weights '{filename}' from {spec.pretrained_id}: {exc}"
                        ) from exc
                    state_dict = torch.load(downloaded_path, map_location="cpu")
                    ckpt_path = Path(downloaded_path)
                elif checkpoint_file:
                    print(f"⚠️ RadImageNet checkpoint file not found: {checkpoint_file}")

            if state_dict is not None:
                state_dict = _remap_radimagenet_state_dict(state_dict)
                missing, unexpected = self.encoder.load_state_dict(
                    state_dict, strict=False
                )
                if missing or unexpected:
                    print(
                        f"⚠️ RadImageNet checkpoint load issues. missing={missing}, unexpected={unexpected}"
                    )
            hidden_size = getattr(self.encoder, "num_features", None) or getattr(
                self.encoder, "fc", None
            )
            if isinstance(hidden_size, nn.Module):
                hidden_size = hidden_size.in_features
            if hidden_size is None:
                raise ValueError(
                    "Unable to infer feature dimension for RadImageNet model"
                )
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported medical foundation family '{spec.family}'")

        head_cfg = kwargs.pop("head", None)
        self.classifier = build_classification_head(
            hidden_size,
            num_classes,
            head_cfg=head_cfg,
            default_dropout=classifier_dropout,
        )

        if spec.family == "biovil":
            mean = spec.image_mean or (0.485, 0.456, 0.406)
            std = spec.image_std or (0.229, 0.224, 0.225)
        elif spec.image_mean and spec.image_std:
            mean, std = spec.image_mean, spec.image_std
        else:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            if spec.pretrained_id and AutoImageProcessor is not None:
                try:
                    processor = AutoImageProcessor.from_pretrained(
                        spec.pretrained_id,
                        trust_remote_code=spec.trust_remote_code,
                        revision=spec.revision,
                    )
                    mean = tuple(
                        float(v) for v in getattr(processor, "image_mean", mean)
                    )
                    std = tuple(float(v) for v in getattr(processor, "image_std", std))
                except Exception:
                    pass
        self.register_buffer("imagenet_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("imagenet_std", IMAGENET_STD.clone(), persistent=False)
        self.register_buffer(
            "model_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "model_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

    # ------------------------------------------------------------------
    # Core forward utilities
    # ------------------------------------------------------------------
    def _convert_inputs(self, images: torch.Tensor) -> torch.Tensor:
        x = images * self.imagenet_std.to(images.device) + self.imagenet_mean.to(
            images.device
        )
        x = (x - self.model_mean.to(images.device)) / self.model_std.to(images.device)
        return x

    def _pool_features(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            if self.pool_type == "mean":
                return hidden.mean(dim=1)
            return hidden[:, 0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError("Unable to pool features from encoder output")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.spec.family == "timm_resnet":
            feats = self.encoder(images)
            if feats.ndim > 2:
                feats = feats.mean(dim=(2, 3))
        elif self.spec.family == "biovil":
            pixel_values = self._convert_inputs(images)
            feats = self.encoder(pixel_values)  # type: ignore[operator]
        else:
            pixel_values = self._convert_inputs(images)
            outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
            feats = self._pool_features(outputs)
        logits = self.classifier(feats)
        return logits

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.spec.family == "timm_resnet":
            feats = self.encoder(images)
            if feats.ndim > 2:
                feats = feats.mean(dim=(2, 3))
            return feats
        pixel_values = self._convert_inputs(images)
        if self.spec.family == "biovil":
            return self.encoder(pixel_values)  # type: ignore[operator]
        outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        return self._pool_features(outputs)

    def freeze_backbone(self) -> None:
        if self.spec.family == "timm_resnet":
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
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
            groups.append(
                {"params": encoder_params, "lr": base_lr, "weight_decay": weight_decay}
            )
        if head_params:
            groups.append(
                {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
            )
        return groups or [
            {"params": self.parameters(), "lr": head_lr, "weight_decay": weight_decay}
        ]

    def step_epoch(self) -> None:
        self.current_epoch += 1
        if (
            self.freeze_backbone_epochs
            and self.current_epoch >= self.freeze_backbone_epochs
        ):
            for param in self.encoder.parameters():
                if not param.requires_grad:
                    param.requires_grad = True


class FoundationMedicalRegistry:
    _registry: Dict[str, FoundationMedicalSpec] = {
        "biomedclip_vit_b16": FoundationMedicalSpec(
            key="biomedclip_vit_b16",
            family="clip",
            pretrained_id="chuhac/BiomedCLIP-vit-bert-hf",
            input_size=(224, 224),
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
            trust_remote_code=True,
        ),
        "biovil_t": FoundationMedicalSpec(
            key="biovil_t",
            family="biovil",
            pretrained_id="microsoft/BiomedVLP-BioViL-T",
            input_size=(448, 448),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            extra_model_kwargs={
                "weights_filename": "biovil_t_image_model_proj_size_128.pt",
            },
        ),
        "radimagenet_resnet50": FoundationMedicalSpec(
            key="radimagenet_resnet50",
            family="timm_resnet",
            pretrained_id="Lab-Rasool/RadImageNet",
            input_size=(224, 224),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            extra_model_kwargs={
                "model_name": "resnet50",
                "weights_filename": "ResNet50.pt",
            },
        ),
    }

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> FoundationMedicalModel:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown foundation_medical model '{key}'. Available: {available}"
            )
        return FoundationMedicalModel(spec, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get_spec(cls, key: str) -> FoundationMedicalSpec:
        spec = cls._registry.get(key)
        if spec is None:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown foundation_medical model '{key}'. Available: {available}"
            )
        return spec


__all__ = [
    "FoundationMedicalModel",
    "FoundationMedicalRegistry",
    "FoundationMedicalSpec",
]
