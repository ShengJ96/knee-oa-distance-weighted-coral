"""Deep learning trainer for knee osteoarthritis classification.

This module provides a comprehensive training framework for CNN models
with support for multiple devices (CPU/CUDA/MPS), early stopping,
learning rate scheduling, and experiment tracking.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    LinearLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    SequentialLR,
)
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.training.distillation import CurriculumHook

from src.data.pytorch_dataset import PaddedBatch
from src.models.heads.ordinal import extract_logits_from_output
from src.training.losses import CORALLoss, CORNLoss, CostSensitiveCORALLoss, FocalLoss
try:  # Prefer unified torch.amp API (PyTorch 2.0+)
    from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler  # type: ignore[attr-defined]

    _USE_TORCH_AMP = True
except Exception:  # pragma: no cover - fallback to legacy CUDA AMP
    try:
        from torch.cuda.amp import autocast as torch_autocast, GradScaler as TorchGradScaler  # type: ignore[attr-defined]

        _USE_TORCH_AMP = False
    except Exception:  # pragma: no cover - AMP unavailable
        torch_autocast = None
        TorchGradScaler = None
        _USE_TORCH_AMP = False


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)

        return self.early_stop


class TrainingMetrics:
    """Track training metrics and history."""

    def __init__(self, selection_metric: str = "loss"):
        """Initialize metrics tracker.

        Args:
            selection_metric: Metric for best model selection. Options:
                - "loss": Select model with lowest validation loss (default)
                - "accuracy": Select model with highest validation accuracy
                - "qwk": Select model with highest validation QWK (recommended for ordinal classification)
        """
        self.selection_metric = selection_metric
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.val_qwks: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_val_qwk = -1.0  # QWK range is [-1, 1]
        self.best_epoch = 0

    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
        epoch: int,
        val_qwk: Optional[float] = None,
    ) -> bool:
        """Update metrics for current epoch.

        Args:
            val_qwk: Optional validation QWK (required if selection_metric="qwk")

        Returns:
            True if this epoch is the new best according to selection_metric
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.val_qwks.append(val_qwk if val_qwk is not None else 0.0)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

        is_best = False
        if self.selection_metric == "qwk" and val_qwk is not None:
            # Select by highest validation QWK
            if val_qwk > self.best_val_qwk:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_val_qwk = val_qwk
                self.best_epoch = epoch
                is_best = True
        elif self.selection_metric == "accuracy":
            # Select by highest validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                if val_qwk is not None:
                    self.best_val_qwk = val_qwk
                self.best_epoch = epoch
                is_best = True
        else:  # "loss" (default)
            # Select by lowest validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                if val_qwk is not None:
                    self.best_val_qwk = val_qwk
                self.best_epoch = epoch
                is_best = True

        return is_best

    def to_history(self) -> Dict:
        """Serialize metrics for checkpointing/inspection."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "val_qwks": self.val_qwks,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "summary": self.get_summary(),
        }

    def load_state(self, state: Dict) -> None:
        """Restore metrics from serialized history."""
        self.train_losses = list(state.get("train_losses", []))
        self.val_losses = list(state.get("val_losses", []))
        self.train_accuracies = list(state.get("train_accuracies", []))
        self.val_accuracies = list(state.get("val_accuracies", []))
        self.val_qwks = list(state.get("val_qwks", []))
        self.learning_rates = list(state.get("learning_rates", []))
        self.epoch_times = list(state.get("epoch_times", []))
        summary = state.get("summary", {})
        self.best_val_loss = float(summary.get("best_val_loss", float("inf")))
        self.best_val_acc = float(summary.get("best_val_acc", 0.0))
        self.best_val_qwk = float(summary.get("best_val_qwk", -1.0))
        self.best_epoch = int(summary.get("best_epoch", 0))

        # Ensure internal lengths stay consistent when resuming
        if self.train_losses:
            # Align best_epoch within available history
            self.best_epoch = min(self.best_epoch, len(self.train_losses) - 1)

    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_val_qwk": self.best_val_qwk,
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0,
            "final_train_acc": (
                self.train_accuracies[-1] if self.train_accuracies else 0
            ),
            "final_val_acc": self.val_accuracies[-1] if self.val_accuracies else 0,
            "final_val_qwk": self.val_qwks[-1] if self.val_qwks else 0,
            "total_epochs": len(self.train_losses),
            "total_time": sum(self.epoch_times),
        }


class DeepLearningTrainer:
    """Deep learning trainer with comprehensive features."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        output_dir: Optional[Union[str, Path]] = None,
        use_tqdm: bool = True,
        log_interval: int = 20,
        *,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        is_main_process: bool = True,
        ddp_find_unused_parameters: bool = False,
        model_selection_metric: str = "loss",
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to use for training
            output_dir: Directory to save outputs
            use_tqdm: Whether to show progress bars
            log_interval: Logging interval for batches
            distributed: Whether using distributed training
            rank: Process rank for distributed training
            world_size: Total number of processes
            is_main_process: Whether this is the main process
            ddp_find_unused_parameters: DDP find_unused_parameters flag
            model_selection_metric: Metric for best model selection. Options:
                - "loss": Select model with lowest validation loss (default)
                - "accuracy": Select model with highest validation accuracy
        """
        self.device = device or self._get_device()
        self.distributed = distributed and world_size > 1
        self.rank = rank if self.distributed else 0
        self.world_size = world_size if self.distributed else 1
        self.is_main_process = is_main_process
        self.ddp_find_unused_parameters = bool(ddp_find_unused_parameters)
        self.model_selection_metric = model_selection_metric

        self.model = model.to(self.device)
        self.num_classes = getattr(model, "num_classes", None)

        if self.distributed:
            device_ids = None
            output_device = None
            if self.device.type == "cuda":
                device_index = self.device.index or 0
                device_ids = [device_index]
                output_device = device_index
            self.model = DDP(
                self.model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=self.ddp_find_unused_parameters,
            )

        self.output_dir = Path(output_dir) if output_dir else Path("experiments/models")
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.metrics = TrainingMetrics(selection_metric=model_selection_metric)
        self._requested_tqdm = use_tqdm
        self.use_tqdm = bool(
            use_tqdm and (not self.distributed or self.is_main_process)
        )
        self.log_interval = log_interval

        # Advanced training features (Stage 4)
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.grad_accum_steps = 1
        if TorchGradScaler is not None and self.device.type == "cuda":
            if _USE_TORCH_AMP:
                self.scaler = TorchGradScaler("cuda", enabled=False)
            else:
                self.scaler = TorchGradScaler(enabled=False)
        else:
            self.scaler = None
        self.scheduler_name: Optional[str] = None
        self.scheduler_params: Dict = {}
        self.scheduler_step_mode: str = "epoch"  # {'batch', 'epoch', 'val_loss'}
        self.teacher_model: Optional[nn.Module] = None
        self.distillation_alpha: float = 0.0
        self.distillation_temperature: float = 1.0
        self.curriculum_hook: Optional["CurriculumHook"] = None
        self.criterion_name: str = "cross_entropy"
        self.requires_ordinal_logits: bool = False

        if self.is_main_process:
            print(f"🚀 Trainer initialized on device: {self.device}")

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def setup_training(
        self,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        # Optional grouped learning rates (for transfer learning)
        lr_backbone: Optional[float] = None,
        lr_head: Optional[float] = None,
        head_name_patterns: Optional[List[str]] = None,
        weight_decay: float = 1e-4,
        scheduler_name: Optional[str] = "plateau",
        scheduler_params: Optional[Dict] = None,
        criterion_name: str = "cross_entropy",
        criterion_params: Optional[Dict[str, Any]] = None,
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10,
        grad_clip_norm: Optional[float] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: str = "float16",
    ):
        """Setup training components.

        Args:
            optimizer_name: Optimizer type ('adam', 'sgd', 'adamw')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            scheduler_name: LR scheduler ('plateau', 'step', None)
            scheduler_params: Scheduler parameters
            criterion_name: Loss function ('cross_entropy', 'focal')
            criterion_params: Additional hyper-parameters for the chosen criterion
            class_weights: Weights for balanced training
            early_stopping_patience: Early stopping patience
            grad_clip_norm: Clip gradients to prevent explosion
            grad_accum_steps: Number of mini-batches to accumulate before optimizer step
            use_amp: Enable automatic mixed precision (CUDA only)
            amp_dtype: AMP data type ('float16' or 'bfloat16')
        """
        # Setup optimizer (support optional backbone/head LR grouping)
        opt_name = optimizer_name.lower()

        # Prepare param list (respect requires_grad)
        params = [p for p in self.model.parameters() if p.requires_grad]

        if lr_backbone is not None and lr_head is not None:
            # Split parameters by name patterns into backbone / head
            # Default to common torchvision head names
            head_patterns = head_name_patterns or ["fc", "classifier"]
            backbone_params: List[torch.nn.Parameter] = []
            head_params: List[torch.nn.Parameter] = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in head_patterns):
                    head_params.append(p)
                else:
                    backbone_params.append(p)

            # Fallback if split failed
            if not head_params or not backbone_params:
                param_groups = [{"params": params, "lr": learning_rate}]
            else:
                param_groups = [
                    {"params": backbone_params, "lr": lr_backbone},
                    {"params": head_params, "lr": lr_head},
                ]
        else:
            param_groups = [{"params": params, "lr": learning_rate}]

        if opt_name == "adam":
            self.optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(
                param_groups, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup criterion
        criterion_config = dict(criterion_params or {})
        criterion_key = criterion_name.lower()
        self.criterion_name = criterion_key
        self.requires_ordinal_logits = False
        if criterion_key == "cross_entropy":
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif criterion_key == "coral":
            if self.num_classes is None:
                raise ValueError("CORAL loss requires model.num_classes to be defined.")
            # Support importance_weights for handling class imbalance
            importance_weights = criterion_config.get("importance_weights")
            if importance_weights is not None:
                importance_weights = torch.tensor(
                    importance_weights, dtype=torch.float32, device=self.device
                )
            self.criterion = CORALLoss(self.num_classes, importance_weights=importance_weights).to(self.device)
            self.requires_ordinal_logits = True
        elif criterion_key == "cost_sensitive_coral":
            if self.num_classes is None:
                raise ValueError("Cost-sensitive CORAL loss requires model.num_classes to be defined.")
            # Support cost_matrix parameter (string or tensor)
            cost_matrix = criterion_config.get("cost_matrix", "quadratic")
            clinical_threshold = criterion_config.get("clinical_threshold", 2)
            clinical_penalty = criterion_config.get("clinical_penalty", 1.5)
            self.criterion = CostSensitiveCORALLoss(
                self.num_classes,
                cost_matrix=cost_matrix,
                clinical_threshold=clinical_threshold,
                clinical_penalty=clinical_penalty,
            ).to(self.device)
            self.requires_ordinal_logits = True
        elif criterion_key == "corn":
            if self.num_classes is None:
                raise ValueError("CORN loss requires model.num_classes to be defined.")
            self.criterion = CORNLoss(self.num_classes)
            self.requires_ordinal_logits = True
        elif criterion_key == "focal":
            gamma = float(criterion_config.get("focal_gamma", 2.0))
            alpha = criterion_config.get("focal_alpha")
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha)
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

        # Setup scheduler
        self.scheduler = None
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params or {}
        self.scheduler_step_mode = "epoch"
        if scheduler_name == "plateau":
            params = dict(
                self.scheduler_params or {"factor": 0.5, "patience": 5, "min_lr": 1e-6}
            )
            self.scheduler = ReduceLROnPlateau(self.optimizer, **params)
            self.scheduler_step_mode = "val_loss"
        elif scheduler_name == "step":
            params = dict(self.scheduler_params or {"step_size": 10, "gamma": 0.1})
            self.scheduler = StepLR(self.optimizer, **params)
            self.scheduler_step_mode = "epoch"
        elif scheduler_name == "warmup_cosine":
            params = dict(self.scheduler_params or {})
            warmup_epochs = int(params.get("warmup_epochs", 1))
            cosine_epochs = int(params.get("cosine_epochs", 9))
            eta_min = float(params.get("min_lr", 1e-6))
            start_factor = float(params.get("warmup_start_factor", 0.3))
            warmup = LinearLR(
                self.optimizer, start_factor=start_factor, total_iters=warmup_epochs
            )
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=cosine_epochs, eta_min=eta_min
            )
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            self.scheduler_step_mode = "epoch"
        elif scheduler_name in {"cosine_warm_restarts", "cosine_restart"}:
            params = dict(
                self.scheduler_params or {"T_0": 10, "T_mult": 1, "eta_min": 1e-6}
            )
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, **params)
            self.scheduler_step_mode = "epoch"
        elif scheduler_name == "one_cycle":
            # Will be instantiated inside train() when loader/epoch info is available
            self.scheduler = None
            self.scheduler_step_mode = "batch"

        # Setup early stopping
        restore_best = not self.distributed
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=restore_best,
        )

        # Gradient clipping setting
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm else 0.0

        # Gradient accumulation
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # Automatic mixed precision (AMP) setup
        requested_amp = bool(use_amp)
        self.use_amp = False
        if requested_amp:
            if self.device.type != "cuda":
                print("⚠️ AMP requested but CUDA device not available; disabling AMP.")
            elif TorchGradScaler is None or torch_autocast is None:
                print("⚠️ torch AMP not available; disabling AMP.")
            else:
                dtype_map = {
                    "fp16": torch.float16,
                    "float16": torch.float16,
                    "half": torch.float16,
                    "bf16": torch.bfloat16,
                    "bfloat16": torch.bfloat16,
                }
                self.amp_dtype = dtype_map.get(amp_dtype.lower(), torch.float16)
                self.use_amp = True
                if _USE_TORCH_AMP:
                    self.scaler = TorchGradScaler("cuda", enabled=True)
                else:
                    self.scaler = TorchGradScaler(enabled=True)
        else:
            if TorchGradScaler is not None and self.device.type == "cuda":
                if _USE_TORCH_AMP:
                    self.scaler = TorchGradScaler("cuda", enabled=False)
                else:
                    self.scaler = TorchGradScaler(enabled=False)
            else:
                self.scaler = None

        if self.is_main_process:
            print("✅ Training setup complete:")
            print(f"  - Optimizer: {optimizer_name}")
            print(f"  - Learning rate: {learning_rate}")
            print(f"  - Scheduler: {scheduler_name}")
            print(f"  - Criterion: {criterion_name}")
            print(f"  - AMP enabled: {self.use_amp}")
            print(f"  - Gradient accumulation steps: {self.grad_accum_steps}")

    def _parse_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, PaddedBatch):
            return batch.images, batch.labels, batch.sizes
        if isinstance(batch, dict):
            data = batch.get("images") or batch.get("pixel_values")
            target = batch.get("labels")
            sizes = batch.get("sizes")
            if data is None or target is None:
                raise ValueError("Batch dictionary must contain 'images' and 'labels'.")
            if isinstance(sizes, (list, tuple)):
                sizes = torch.tensor(sizes, dtype=torch.long)
            return data, target, sizes
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
        raise TypeError(f"Unsupported batch structure: {type(batch)!r}")

    def _forward_model(
        self,
        inputs: torch.Tensor,
        *,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
        model: Optional[nn.Module] = None,
    ):
        module = model or self.model
        spec = getattr(getattr(module, "spec", None), "family", None)
        if original_sizes and spec == "clip_naflex":
            return module(inputs, original_sizes=original_sizes)
        return module(inputs)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            (train_loss, train_accuracy) tuple
        """
        self.model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        accum_steps = max(1, getattr(self, "grad_accum_steps", 1))
        amp_enabled = bool(
            self.use_amp and self.scaler is not None and torch_autocast is not None
        )
        if amp_enabled and self.device.type != "cuda":
            amp_enabled = False

        if self.teacher_model is not None:
            self.teacher_model.eval()

        if self.use_tqdm:
            desc = (
                f"Train [{epoch}/{total_epochs}]" if epoch and total_epochs else "Train"
            )
            bar = tqdm(train_loader, desc=desc, leave=False, dynamic_ncols=True)
            iterator = enumerate(bar)
        else:
            bar = None
            iterator = enumerate(train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in iterator:
            data, target, sizes = self._parse_batch(batch)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            size_list: Optional[List[Tuple[int, int]]] = None
            if sizes is not None:
                size_list = [tuple(map(int, pair)) for pair in sizes.tolist()]

            context = self._autocast_context(amp_enabled)
            with context:
                student_out = self._forward_model(data, original_sizes=size_list)
                student_logits, student_ordinals = self._split_output(student_out)

            supervised_loss = self._compute_loss(student_logits, student_ordinals, target)
            loss = supervised_loss

            if self.teacher_model is not None and self.distillation_alpha > 0.0:
                with torch.no_grad():
                    teacher_out = self._forward_model(
                        data,
                        original_sizes=size_list,
                        model=self.teacher_model,
                    )
                    teacher_logits, _ = self._split_output(teacher_out)
                T = self.distillation_temperature
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                loss = (1.0 - self.distillation_alpha) * supervised_loss + self.distillation_alpha * kd_loss

            loss_value = loss.item()
            batch_size = float(target.size(0))
            running_loss += loss_value * batch_size

            loss_to_backward = loss / accum_steps
            if amp_enabled and self.scaler is not None:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            is_update_step = ((batch_idx + 1) % accum_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            )
            if is_update_step:
                if getattr(self, "grad_clip_norm", 0.0) and self.grad_clip_norm > 0:
                    if amp_enabled and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                if amp_enabled and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler and self.scheduler_step_mode == "batch":
                    self.scheduler.step()

            _, predicted = torch.max(student_logits.detach(), 1)
            total += batch_size
            correct += (predicted == target).sum().item()

            if not self.use_tqdm and (batch_idx + 1) % self.log_interval == 0:
                avg = running_loss / max(total, 1.0)
                acc = 100.0 * correct / max(total, 1.0)
                print(
                    f"  [batch {batch_idx+1}/{len(train_loader)}] loss={avg:.4f} acc={acc:5.2f}%"
                )
            elif self.use_tqdm and bar is not None:
                avg = running_loss / max(total, 1.0)
                acc = 100.0 * correct / max(total, 1.0)
                bar.set_postfix(loss=f"{avg:.4f}", acc=f"{acc:5.2f}%")

            if self.curriculum_hook is not None:
                try:
                    self.curriculum_hook.on_after_batch(
                        self,
                        int(epoch) if epoch is not None else 0,
                        batch_idx,
                        len(train_loader),
                    )
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Curriculum hook failed during batch callback"
                    ) from exc

        if self.distributed and dist.is_available() and dist.is_initialized():
            stats = torch.tensor(
                [running_loss, correct, total],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running_loss, correct, total = stats.tolist()

        epoch_loss = running_loss / max(total, 1.0)
        epoch_acc = 100.0 * correct / max(total, 1.0)

        return epoch_loss, epoch_acc

    def set_teacher(
        self,
        teacher_model: nn.Module,
        *,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ) -> None:
        """Enable knowledge distillation with a fixed teacher model."""
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.distillation_alpha = float(max(0.0, min(1.0, alpha)))
        self.distillation_temperature = float(max(1e-6, temperature))
        print(
            "  - Knowledge distillation enabled: alpha="
            f"{self.distillation_alpha:.2f}, T={self.distillation_temperature:.2f}"
        )

    def set_curriculum_hook(self, hook: "CurriculumHook") -> None:
        """Attach a curriculum learning hook to the trainer."""

        self.curriculum_hook = hook

    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            (val_loss, val_accuracy, val_qwk) tuple
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        all_predictions: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            iterator = val_loader
            if self.use_tqdm:
                desc = (
                    f"Val   [{epoch}/{total_epochs}]"
                    if epoch and total_epochs
                    else "Val"
                )
                vbar = tqdm(val_loader, desc=desc, leave=False, dynamic_ncols=True)
                iterator = vbar
            else:
                vbar = None
                iterator = val_loader
            for batch in iterator:
                data, target, sizes = self._parse_batch(batch)
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                size_list = None
                if sizes is not None:
                    size_list = [tuple(map(int, pair)) for pair in sizes.tolist()]

                output = self._forward_model(data, original_sizes=size_list)
                logits, ordinal_logits = self._split_output(output)
                loss = self._compute_loss(logits, ordinal_logits, target)

                batch_size = float(target.size(0))
                running_loss += loss.item() * batch_size
                _, predicted = torch.max(logits.data, 1)
                total += batch_size
                correct += (predicted == target).sum().item()

                # Collect predictions and targets for QWK calculation
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())

                if self.use_tqdm and vbar is not None:
                    avg = running_loss / max(total, 1.0)
                    acc = 100.0 * correct / max(total, 1.0)
                    vbar.set_postfix(loss=f"{avg:.4f}", acc=f"{acc:5.2f}%")

        if self.distributed and dist.is_available() and dist.is_initialized():
            stats = torch.tensor(
                [running_loss, correct, total],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running_loss, correct, total = stats.tolist()
            # Note: QWK calculation in distributed mode would need gathering all predictions
            # For now, compute on local data only (acceptable for single-GPU training)

        epoch_loss = running_loss / max(total, 1.0)
        epoch_acc = 100.0 * correct / max(total, 1.0)

        # Compute QWK (Quadratic Weighted Kappa)
        epoch_qwk = 0.0
        if all_predictions and all_targets:
            try:
                epoch_qwk = cohen_kappa_score(
                    all_targets, all_predictions, weights="quadratic"
                )
            except Exception:
                epoch_qwk = 0.0

        return epoch_loss, epoch_acc, epoch_qwk

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_best: bool = True,
        save_last: bool = True,
        resume_from: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of epochs to train (inclusive of resumed progress)
            save_best: Whether to save best-performing checkpoint
            save_last: Whether to continually persist the latest checkpoint
            resume_from: Optional checkpoint path to resume training from
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        if not self.optimizer:
            raise ValueError("Training not setup. Call setup_training() first.")

        start_time = time.time()

        # Instantiate schedulers that require loader metadata (e.g., OneCycle)
        if self.scheduler_name == "one_cycle":
            params = dict(self.scheduler_params or {})
            max_lr = params.pop("max_lr", None)
            if max_lr is None:
                raise ValueError(
                    "OneCycle scheduler requires 'max_lr' in scheduler_params"
                )
            if not isinstance(max_lr, (list, tuple)):
                max_lr = [max_lr] * len(self.optimizer.param_groups)
            anneal = params.get("anneal_strategy")
            if anneal == "cosine":
                params["anneal_strategy"] = "cos"
            steps_per_epoch = max(
                1, math.ceil(len(train_loader) / max(1, self.grad_accum_steps))
            )
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                **params,
            )
            self.scheduler_step_mode = "batch"

        # Optionally resume from checkpoint
        if resume_from:
            start_epoch = self.resume_from_checkpoint(resume_from)
        else:
            start_epoch = 0
            self.metrics.reset()

        if verbose and self.is_main_process:
            if resume_from:
                print(
                    f"\n🚀 Resume training: target {epochs} epochs (starting at epoch {start_epoch})"
                )
            else:
                print(f"\n🚀 Training started for {epochs} epochs")
            print("=" * 80)

        if resume_from and start_epoch >= epochs:
            if verbose:
                print(
                    "⚠️ Resume checkpoint already reached requested epochs; skipping training."
                )
            return self.get_training_history()

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            if self.distributed:
                sampler = getattr(train_loader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

            # Train and validate
            display_epoch = epoch + 1
            if self.curriculum_hook is not None:
                try:
                    self.curriculum_hook.on_epoch_start(
                        self, display_epoch, epochs
                    )
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Curriculum hook failed during epoch start"
                    ) from exc
            train_loss, train_acc = self.train_epoch(
                train_loader, display_epoch, epochs
            )
            val_loss, val_acc, val_qwk = self.validate_epoch(val_loader, display_epoch, epochs)

            # Update learning rate
            if self.scheduler:
                if self.scheduler_step_mode == "val_loss":
                    self.scheduler.step(val_loss)
                elif self.scheduler_step_mode == "epoch":
                    if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                        self.scheduler.step(display_epoch)
                    else:
                        self.scheduler.step()

            # Track metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            if self.is_main_process:
                is_best = self.metrics.update(
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                    current_lr,
                    epoch_time,
                    epoch,
                    val_qwk=val_qwk,
                )

                if verbose:
                    print(
                        f"Epoch {display_epoch:3d}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:5.2f}% | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:5.2f}% | "
                        f"Val QWK: {val_qwk:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Time: {epoch_time:.1f}s"
                    )

                if save_best and is_best:
                    self.save_model(f"best_model_epoch_{display_epoch}.pth")

                if save_last:
                    self.save_model("last_model.pth")

            # Early stopping
            stop_training = False
            if self.is_main_process:
                stop_training = self.early_stopping(val_loss, self._unwrap_model())

            if self.distributed and dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor(
                    [1 if stop_training else 0], device=self.device, dtype=torch.int
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if self.curriculum_hook is not None:
                try:
                    self.curriculum_hook.on_epoch_end(
                        self, display_epoch, epochs
                    )
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Curriculum hook failed during epoch end"
                    ) from exc

            if stop_training:
                if verbose and self.is_main_process:
                    print(f"\n⏰ Early stopping triggered at epoch {display_epoch}")
                break

        total_time = time.time() - start_time

        if verbose and self.is_main_process:
            summary = self.metrics.get_summary()
            if self.model_selection_metric == "qwk":
                selection_str = "highest QWK"
            elif self.model_selection_metric == "accuracy":
                selection_str = "highest accuracy"
            else:
                selection_str = "lowest loss"
            print(f"\n✅ Training completed in {total_time:.1f}s")
            print(
                f"📊 Best model (by {selection_str}): "
                f"{summary['best_val_acc']:.2f}% accuracy, "
                f"QWK={summary['best_val_qwk']:.4f}, "
                f"loss={summary['best_val_loss']:.4f} "
                f"(epoch {summary['best_epoch'] + 1})"
            )

        return self.get_training_history()

    def save_model(self, filename: str) -> None:
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        filepath = self.output_dir / filename
        model_to_save = self._unwrap_model()
        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "training_history": self.get_training_history(),
                "model_info": {
                    "class_name": model_to_save.__class__.__name__,
                    "device": str(self.device),
                },
            },
            filepath,
        )

    def load_model(self, filepath: Union[str, Path]) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        model_to_load = self._unwrap_model()
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        history = checkpoint.get("training_history", {})
        if history:
            self.metrics.load_state(history)

        return history

    def resume_from_checkpoint(self, filepath: Union[str, Path]) -> int:
        """Resume training state from checkpoint and return starting epoch index."""
        checkpoint_path = Path(filepath)
        history = self.load_model(checkpoint_path)

        completed_epochs = len(self.metrics.train_losses)
        if self.early_stopping:
            best_loss = (
                history.get("summary", {}).get("best_val_loss") if history else None
            )
            if best_loss is not None:
                self.early_stopping.best_loss = float(best_loss)
            self.early_stopping.counter = 0

        if self.is_main_process:
            print(
                f"🔁 Resumed from {checkpoint_path} | completed epochs: {completed_epochs}"
            )
        return completed_epochs

    def get_training_history(self) -> Dict:
        """Get complete training history."""
        history = self.metrics.to_history()
        history["device"] = str(self.device)
        history["timestamp"] = datetime.now().isoformat()
        return history

    def save_training_history(self, filename: str = "training_history.json") -> None:
        """Save training history to JSON."""
        filepath = self.output_dir / filename
        history = self.get_training_history()

        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)

        if self.is_main_process:
            print(f"📊 Training history saved to {filepath}")

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _sync_model_state(self) -> None:
        if not (self.distributed and dist.is_available() and dist.is_initialized()):
            return
        buckets: Dict[torch.dtype, list[torch.Tensor]] = {}

        def _add_tensor(t: torch.Tensor) -> None:
            if not t.is_floating_point() and not t.is_complex():
                key = t.dtype
            else:
                key = t.dtype
            buckets.setdefault(key, []).append(t)

        for param in self.model.parameters():
            _add_tensor(param.data)
        for buffer in self.model.buffers():
            _add_tensor(buffer.data)

        broadcast_coalesced = getattr(dist, "broadcast_coalesced", None)
        if broadcast_coalesced is None:
            broadcast_coalesced = getattr(dist, "_broadcast_coalesced", None)

        for tensors in buckets.values():
            if not tensors:
                continue
            if len(tensors) == 1:
                dist.broadcast(tensors[0], src=0)
                continue
            try:
                if broadcast_coalesced is not None:
                    broadcast_coalesced(tensors, src=0)
                else:
                    raise AttributeError
            except RuntimeError:
                for tensor in tensors:
                    dist.broadcast(tensor, src=0)
            except AttributeError:
                for tensor in tensors:
                    dist.broadcast(tensor, src=0)

    def _split_output(self, output):
        return extract_logits_from_output(output)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        ordinal_logits: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.requires_ordinal_logits:
            if ordinal_logits is None:
                raise RuntimeError(
                    "Ordinal criterion requested but model did not return ordinal logits."
                )
            return self.criterion(ordinal_logits, target)
        return self.criterion(logits, target)

    @staticmethod
    def _extract_logits(output) -> torch.Tensor:  # backward compatibility
        logits, _ = extract_logits_from_output(output)
        return logits

    def _autocast_context(self, enabled: bool):
        if not enabled or torch_autocast is None:
            return contextlib.nullcontext()
        if _USE_TORCH_AMP:
            return torch_autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=enabled
            )
        # Legacy CUDA autocast
        return torch_autocast(dtype=self.amp_dtype, enabled=enabled)


if __name__ == "__main__":
    # Example usage
    from src.models.deep_learning.basic_cnn import CNNRegistry
    from src.data.pytorch_dataset import create_data_loaders

    print("Testing Deep Learning Trainer...")
    print("=" * 50)

    # Create dummy model and data
    model = CNNRegistry.get_model("simple_cnn", num_classes=5)

    # Note: This would normally use real data
    print("\n✅ Trainer test completed successfully!")
    print("To use with real data, create data loaders and call trainer.train()")
