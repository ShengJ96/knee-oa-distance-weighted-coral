"""Visualization helpers for Stage 4 attention and attribution maps."""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.pytorch_dataset import PaddedBatch
from src.models.heads.ordinal import extract_logits_from_output

# Mean/std used for ImageNet-pretrained models (dataset normalized accordingly)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _model_supports_original_sizes(model: torch.nn.Module) -> bool:
    return getattr(getattr(model, "spec", None), "family", "") == "clip_naflex"


def _unwrap_batch(
    batch,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, PaddedBatch):
        return batch.images, batch.labels, batch.sizes
    if isinstance(batch, dict):
        images = batch.get("images") or batch.get("pixel_values")
        labels = batch.get("labels")
        sizes = batch.get("sizes")
        if images is None or labels is None:
            raise ValueError("Batch dictionary must include 'images' and 'labels'.")
        if isinstance(sizes, (list, tuple)):
            sizes = torch.tensor(sizes, dtype=torch.long)
        return images, labels, sizes
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise TypeError(f"Unsupported batch structure: {type(batch)!r}")


@dataclass
class AttentionVizConfig:
    """Configuration for attention visualization generation."""

    samples_per_class: int = 2
    rollout: bool = False
    colormap: str = "jet"
    seed: Optional[int] = None


class ResultVisualizer:
    """Generate attention visualizations (Grad-CAM / attention rollout)."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Optional[torch.device] = None,
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.model = model
        self.device = device or getattr(model, "device", None) or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_predictions(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Utility to collect model predictions (logits argmax)."""
        probs: List[int] = []
        supports_sizes = _model_supports_original_sizes(self.model)
        for batch in dataloader:
            images, labels, sizes = _unwrap_batch(batch)
            images = images.to(self.device)
            size_list = None
            if supports_sizes and sizes is not None:
                size_list = [tuple(map(int, pair)) for pair in sizes.tolist()]
            logits = self._forward_logits(images, original_sizes=size_list)
            probs.extend(torch.argmax(logits, dim=1).cpu().tolist())
        return np.asarray(probs, dtype=np.int64)

    def generate_attention_maps(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Path | str,
        config: Optional[AttentionVizConfig] = None,
    ) -> Dict[str, List[Path]]:
        """Generate Grad-CAM or attention rollout visualizations.

        Args:
            dataloader: Data loader yielding (image_tensor, label)
            output_dir: Directory to save generated images
            config: AttentionVizConfig controlling rollout/colormap/limits

        Returns:
            Mapping from class name to list of saved file paths
        """
        cfg = config or AttentionVizConfig()
        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples_per_class = max(1, int(cfg.samples_per_class))
        collected: Dict[int, int] = {}
        saved_paths: Dict[str, List[Path]] = {}

        is_transformer = _is_vision_transformer(self.model)
        gradcam_layer = _infer_gradcam_layer(self.model)

        supports_sizes = _model_supports_original_sizes(self.model)

        for batch in dataloader:
            images, labels, sizes = _unwrap_batch(batch)
            batch_size = images.size(0)
            for idx in range(batch_size):
                label = int(labels[idx])
                if collected.get(label, 0) >= samples_per_class:
                    continue
                h, w = (
                    tuple(map(int, sizes[idx].tolist()))
                    if supports_sizes and sizes is not None
                    else (images.shape[-2], images.shape[-1])
                )
                image_tensor = images[idx: idx + 1, :, :h, :w].to(self.device)
                target_class = label

                original_size_arg = [(h, w)] if supports_sizes else None

                if is_transformer and cfg.rollout:
                    try:
                        heatmap = self._attention_rollout(image_tensor)
                    except RuntimeError:
                        print(
                            "⚠️ Attention rollout unsupported for this architecture; falling back to Grad-CAM."
                        )
                        try:
                            heatmap = self._gradcam(
                                image_tensor, target_class, gradcam_layer, original_sizes=original_size_arg
                            )
                        except RuntimeError:
                            print(
                                "⚠️ Grad-CAM could not infer a target layer; skipping sample."
                            )
                            continue
                else:
                    try:
                        heatmap = self._gradcam(
                            image_tensor, target_class, gradcam_layer, original_sizes=original_size_arg
                        )
                    except RuntimeError:
                        print(
                            "⚠️ Grad-CAM could not infer a target layer; skipping sample."
                        )
                        continue

                class_name = self._class_name(label)
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                overlay_path = class_dir / f"{class_name}_sample{collected.get(label, 0):02d}.png"
                heatmap_path = class_dir / f"{class_name}_sample{collected.get(label, 0):02d}_heatmap.png"

                base_image = _tensor_to_pil(images[idx, :, :h, :w])
                overlay = _overlay_heatmap(base_image, heatmap, cfg.colormap)
                heatmap_image = _heatmap_to_image(heatmap, cfg.colormap).resize(base_image.size, resample=Image.BILINEAR)

                heatmap_image.save(heatmap_path)
                overlay.save(overlay_path)

                saved_paths.setdefault(class_name, []).append(overlay_path)
                collected[label] = collected.get(label, 0) + 1

                if all(count >= samples_per_class for count in collected.values()) and len(collected) >= self._num_classes():
                    return saved_paths

        return saved_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _forward_logits(
        self,
        batch: torch.Tensor,
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if original_sizes and _model_supports_original_sizes(self.model):
                outputs = self.model(batch, original_sizes=original_sizes)
            else:
                outputs = self.model(batch)
            return _extract_logits(outputs)

    def _gradcam(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        target_layer: Optional[torch.nn.Module],
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        model = self.model
        was_training = model.training
        model.eval()

        if target_layer is None:
            raise RuntimeError("Grad-CAM could not infer a target layer for this model.")

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_, __, output):
            activations.append(output.detach())

        def backward_hook(_, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        try:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            with torch.enable_grad():
                input_tensor = image_tensor.requires_grad_(True)
                if original_sizes and _model_supports_original_sizes(model):
                    logits = model(input_tensor, original_sizes=original_sizes)
                else:
                    logits = model(input_tensor)
                logits = _extract_logits(logits)
                score = logits[:, target_class].sum()
                score.backward()

            if not activations or not gradients:
                raise RuntimeError("Failed to capture activations/gradients for Grad-CAM")

            act = activations[0]
            grad = gradients[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=False))
            cam = cam.squeeze(0)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
            heatmap = cam.squeeze().cpu().numpy()
            heatmap = _normalize_heatmap(heatmap)
        finally:
            handle_f.remove()
            handle_b.remove()
            if was_training:
                model.train()

        return heatmap

    def _attention_rollout(self, image_tensor: torch.Tensor) -> np.ndarray:
        model = getattr(self.model, "model", None)
        if model is None:
            raise RuntimeError("Vision transformer wrapper missing `.model` attribute for attention rollout")

        config = getattr(model, "config", None)
        if config is None:
            raise RuntimeError("Transformer model is missing a config for attention rollout")

        original_flag = getattr(config, "output_attentions", False)
        original_impl = getattr(config, "attn_implementation", None)

        reset_impl = None
        if original_impl not in (None, "eager"):
            set_impl = getattr(model, "set_attn_implementation", None)
            if callable(set_impl):
                try:
                    set_impl("eager")
                    reset_impl = original_impl
                except ValueError:
                    pass
            else:
                try:
                    config.attn_implementation = "eager"
                    reset_impl = original_impl
                except ValueError:
                    pass

        config.output_attentions = True

        try:
            with torch.no_grad():
                outputs = model(
                    pixel_values=image_tensor,
                    output_attentions=True,
                    return_dict=True,
                )
        finally:
            if reset_impl is not None:
                set_impl = getattr(model, "set_attn_implementation", None)
                if callable(set_impl):
                    try:
                        set_impl(reset_impl)
                    except ValueError:
                        pass
                else:
                    try:
                        config.attn_implementation = reset_impl
                    except ValueError:
                        pass
            config.output_attentions = original_flag

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError("Transformer model did not return attentions")

        try:
            attn = torch.stack(attentions)  # (layers, batch, heads, tokens, tokens)
        except RuntimeError as exc:
            raise RuntimeError(
                "Attention rollout is not supported for this transformer (windowed or hierarchical attention)."
            ) from exc
        attn = attn.mean(dim=2)  # average heads
        residual = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0).unsqueeze(0)
        attn = attn + residual
        attn = attn / attn.sum(dim=-1, keepdim=True)

        rollout = attn[0]  # assume batch size 1
        for layer in range(1, rollout.size(0)):
            rollout = torch.bmm(attn[layer], rollout)

        mask = rollout[:, 0, 1:]  # remove CLS token
        tokens = mask.shape[-1]
        spatial_dim = int(math.sqrt(tokens))
        heatmap = mask.reshape(-1, spatial_dim, spatial_dim)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap.squeeze(0).squeeze(0).cpu().numpy()
        heatmap = _normalize_heatmap(heatmap)
        return heatmap

    def _class_name(self, label: int) -> str:
        if self.class_names and 0 <= label < len(self.class_names):
            return str(self.class_names[label])
        return f"class_{label}"

    def _num_classes(self) -> int:
        if self.class_names:
            return len(self.class_names)
        return getattr(self.model, "num_classes", 5)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max() + 1e-8
    heatmap = heatmap / denom
    return np.clip(heatmap, 0.0, 1.0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clone()
    tensor = tensor * _IMAGENET_STD + _IMAGENET_MEAN
    tensor = tensor.clamp(0, 1)
    array = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, colormap: str) -> Image.Image:
    heatmap = _normalize_heatmap(heatmap)
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[..., :3]
    colored_img = Image.fromarray((colored * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    overlay = Image.blend(image.convert("RGBA"), colored_img.convert("RGBA"), alpha=0.5)
    return overlay.convert("RGB")


def _heatmap_to_image(heatmap: np.ndarray, colormap: str) -> Image.Image:
    heatmap = _normalize_heatmap(heatmap)
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def _infer_gradcam_layer(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    candidate = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            candidate = module
    if candidate is not None:
        return candidate
    # Swin / HF models: patch embeddings expose `projection` conv inside SwinPatchEmbeddings
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        proj = getattr(patch_embed, "projection", None)
        if isinstance(proj, torch.nn.Conv2d):
            return proj
    # If wrapper holds actual model under `.model`
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        return _infer_gradcam_layer(inner)
    return None


def _is_vision_transformer(model: torch.nn.Module) -> bool:
    spec = getattr(model, "spec", None)
    architecture = getattr(spec, "architecture", None)
    if architecture in {"vit", "swin"}:
        return True
    # HuggingFace models expose encoder blocks; fallback check
    return hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "num_hidden_layers")


def _extract_logits(output) -> torch.Tensor:
    logits, _ = extract_logits_from_output(output)
    return logits
