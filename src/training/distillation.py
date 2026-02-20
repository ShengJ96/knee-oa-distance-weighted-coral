"""Utility helpers for knowledge distillation and curriculum hooks.

This module centralises the logic for enabling teacher-student training and
exposes lightweight curriculum learning hooks that can be plugged into the
``DeepLearningTrainer`` without introducing hard dependencies.  The helpers are
intentionally minimal so that Stage 5 experiments can extend them without
rewriting the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Knowledge distillation configuration helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DistillationConfig:
    """Configuration container for knowledge distillation.

    Attributes mirror the YAML structure used under ``knowledge_distillation``
    sections in experiment configs.
    """

    enable: bool = False
    alpha: float = 0.5
    temperature: float = 2.0
    teacher_model: Optional[Dict[str, Any]] = None
    teacher_checkpoint: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DistillationConfig":
        data = dict(data or {})
        enable = bool(data.get("enable", False))
        alpha = float(data.get("alpha", 0.5))
        temperature = float(data.get("temperature", 2.0))
        teacher_checkpoint = data.get("teacher_checkpoint")
        if teacher_checkpoint is not None:
            teacher_checkpoint = Path(teacher_checkpoint)
        teacher_model = data.get("teacher_model")
        return cls(
            enable=enable,
            alpha=float(max(0.0, min(1.0, alpha))),
            temperature=float(max(1e-6, temperature)),
            teacher_model=teacher_model,
            teacher_checkpoint=teacher_checkpoint,
        )

    def should_enable(self) -> bool:
        """Return ``True`` if distillation should be activated."""

        return self.enable and self.teacher_model is not None


class BuildModelFn(Protocol):
    """Callable signature for model factory helpers used in Stage 4 configs."""

    def __call__(self, model_cfg: Dict[str, Any]) -> nn.Module:  # pragma: no cover - typing
        ...


def build_teacher_from_config(
    cfg: DistillationConfig,
    build_fn: BuildModelFn,
    device: torch.device,
) -> nn.Module:
    """Instantiate and (optionally) restore a teacher model.

    Args:
        cfg: Parsed distillation configuration.
        build_fn: Factory that maps ``teacher_model`` dictionaries to modules.
        device: Target device for the teacher network.

    Returns:
        Initialised teacher model placed on ``device``.

    Raises:
        ValueError: If ``teacher_model`` details are missing while distillation
            is enabled.
        FileNotFoundError: When a checkpoint path is provided but does not
            exist.
    """

    if not cfg.teacher_model:
        raise ValueError("Teacher model configuration is required when distillation is enabled.")

    teacher = build_fn(cfg.teacher_model)
    teacher.to(device)

    if cfg.teacher_checkpoint:
        path = cfg.teacher_checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device)
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        if state_dict is None:
            state_dict = checkpoint
        teacher.load_state_dict(state_dict)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    return teacher


# ---------------------------------------------------------------------------
# Curriculum learning hook scaffolding
# ---------------------------------------------------------------------------


class CurriculumHook(Protocol):
    """Interface for curriculum learning callbacks."""

    def on_epoch_start(
        self,
        trainer: "TrainerProtocol",
        epoch: int,
        total_epochs: int,
    ) -> None:
        ...

    def on_after_batch(
        self,
        trainer: "TrainerProtocol",
        epoch: int,
        batch_index: int,
        total_batches: int,
    ) -> None:
        ...

    def on_epoch_end(
        self,
        trainer: "TrainerProtocol",
        epoch: int,
        total_epochs: int,
    ) -> None:
        ...


class TrainerProtocol(Protocol):  # pragma: no cover - typing helper
    grad_clip_norm: float
    device: torch.device


@dataclass(slots=True)
class CurriculumStep:
    """Single curriculum adjustment rule."""

    epoch: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CurriculumStep":
        return cls(epoch=int(payload["epoch"]), attributes=dict(payload.get("attributes", {})))


class AttributeCurriculumHook:
    """Updates ``trainer`` attributes at predetermined epochs.

    This is a lightweight hook meant to cover simple curriculum strategies such
    as gradually unfreezing layers, changing augmentation strength flags or
    updating optimisation hyper-parameters.
    """

    def __init__(self, steps: Iterable[CurriculumStep]):
        self._steps = {step.epoch: step for step in steps}

    def on_epoch_start(
        self,
        trainer: TrainerProtocol,
        epoch: int,
        total_epochs: int,
    ) -> None:
        step = self._steps.get(epoch)
        if not step:
            return
        for key, value in step.attributes.items():
            if hasattr(trainer, key):
                setattr(trainer, key, value)

    def on_after_batch(
        self,
        trainer: TrainerProtocol,
        epoch: int,
        batch_index: int,
        total_batches: int,
    ) -> None:
        return

    def on_epoch_end(
        self,
        trainer: TrainerProtocol,
        epoch: int,
        total_epochs: int,
    ) -> None:
        return


def build_curriculum_hook(config: Optional[Dict[str, Any]]) -> Optional[CurriculumHook]:
    """Factory that converts configuration dictionaries into curriculum hooks."""

    if not config or not config.get("enable"):
        return None

    steps_cfg = config.get("attribute_schedule", [])
    steps = [CurriculumStep.from_dict(item) for item in steps_cfg]
    if steps:
        return AttributeCurriculumHook(steps)
    return None


__all__ = [
    "DistillationConfig",
    "build_teacher_from_config",
    "build_curriculum_hook",
    "CurriculumHook",
    "AttributeCurriculumHook",
    "CurriculumStep",
]

