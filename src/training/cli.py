"""Command-line interface for Stage 4 deep learning experiments."""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

import click
import numpy as np
import torch
import torch.distributed as dist
import yaml

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.pytorch_dataset import (
    create_data_loaders,
    create_multi_source_data_loaders,
    pad_image_collate,
    PaddedBatch,
)
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualization import AttentionVizConfig, ResultVisualizer
from src.models.deep_learning import (
    AdvancedCNNRegistry,
    CNNRegistry,
    FoundationGeneralRegistry,
    FoundationMedicalRegistry,
    TransferLearningRegistry,
    VisionTransformerRegistry,
)
from src.models.heads.ordinal import extract_logits_from_output
from src.training.dl_trainer import DeepLearningTrainer
from src.training.distillation import (
    DistillationConfig,
    build_curriculum_hook,
    build_teacher_from_config,
)

_MISSING = object()


def _parse_target_size(value: Any) -> Optional[Tuple[int, int]]:
    """Normalize target_size config values to Tuple[int, int] or None."""

    if value is _MISSING:
        return (224, 224)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"target_size must have length 2, got {value!r}")
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return (int(value), int(value))
    raise ValueError(f"Unsupported target_size value {value!r}. Use [H, W] or null.")


def _requires_variable_image_collate(model_cfg: Dict[str, Any]) -> bool:
    registry = (model_cfg.get("registry") or "").lower()
    key = model_cfg.get("key")
    if registry == "foundation_general" and key:
        try:
            spec = FoundationGeneralRegistry.get_spec(key)
        except KeyError:
            return False
        return getattr(spec, "family", "") == "clip_naflex"
    return False


def _model_supports_original_sizes(model: torch.nn.Module) -> bool:
    return getattr(getattr(model, "spec", None), "family", "") == "clip_naflex"


def _unwrap_batch(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, PaddedBatch):
        return batch.images, batch.labels, batch.sizes
    if isinstance(batch, dict):
        images = batch.get("images") or batch.get("pixel_values")
        labels = batch.get("labels")
        sizes = batch.get("sizes")
        if images is None or labels is None:
            raise ValueError("Batch dict must contain 'images' and 'labels'.")
        if isinstance(sizes, (list, tuple)):
            sizes = torch.tensor(sizes, dtype=torch.long)
        return images, labels, sizes
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise TypeError(f"Unsupported batch structure encountered: {type(batch)!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(
    preferred: str | None = None, local_rank: Optional[int] = None
) -> torch.device:
    if preferred:
        if preferred.startswith("cuda") and torch.cuda.is_available():
            if ":" in preferred:
                return torch.device(preferred)
            if local_rank is not None:
                return torch.device(f"cuda:{local_rank}")
        return torch.device(preferred)
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prune_checkpoints(output_dir: Path, best_path: Path, keep_last: bool = True) -> None:
    """
    Keep only the best checkpoint (and optionally last_model.pth) to save disk.
    """
    if not output_dir.exists():
        return
    best_name = Path(best_path).name
    for ckpt in output_dir.glob("best_model_epoch_*.pth"):
        if ckpt.name != best_name:
            ckpt.unlink(missing_ok=True)
    if not keep_last:
        last_ckpt = output_dir / "last_model.pth"
        if last_ckpt.exists():
            last_ckpt.unlink(missing_ok=True)


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "on", "distributed", "ddp", "always"}:
            return True
        if lowered in {"false", "no", "n", "off", "never"}:
            return False
    return None


def _init_distributed(backend: str) -> bool:
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _gather_variable_length_tensor(
    tensor: torch.Tensor, world_size: int
) -> torch.Tensor:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return tensor

    device = tensor.device
    length_tensor = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)
    lengths = [torch.zeros_like(length_tensor) for _ in range(world_size)]
    dist.all_gather(lengths, length_tensor)
    max_len = int(torch.max(torch.stack(lengths)).item())

    if tensor.size(0) < max_len:
        pad_shape = (max_len - tensor.size(0),) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    slices = []
    for buf, ln in zip(gathered, lengths):
        slices.append(buf[: int(ln.item())])

    return torch.cat(slices, dim=0)


def _normalize_dataset_name(root: str | Path | None) -> str:
    if not root:
        return "unknown"
    path = Path(root)
    if path.name.lower().startswith("set_"):
        return path.name.lower()
    for part in path.parts[::-1]:
        if part.lower().startswith("set_"):
            return part.lower()
    return path.name.lower()


def _evaluate_single_dataset(
    cfg: Dict[str, Any],
    dataset_spec: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    collate_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    data_cfg = cfg.get("data", {})
    augmentation = data_cfg.get("augmentation", {})

    root = dataset_spec.get("root")
    if not root:
        raise ValueError("Multi-source dataset spec must include 'root'.")

    single_spec = deepcopy(dataset_spec)
    single_spec.setdefault("limit_per_class", data_cfg.get("limit_per_class"))
    single_spec.setdefault(
        "limit_per_class_val", data_cfg.get("limit_per_class_val")
    )
    single_spec.setdefault(
        "limit_per_class_test", data_cfg.get("limit_per_class_test")
    )
    single_spec.setdefault("medical_variant", data_cfg.get("medical_variant", "none"))

    target_size_cfg = data_cfg.get("target_size", _MISSING)
    _, _, test_loader = create_data_loaders(
        data_root=single_spec["root"],
        batch_size=data_cfg.get("batch_size", 32),
        target_size=_parse_target_size(target_size_cfg),
        num_workers=data_cfg.get("num_workers", 4),
        augment_train=augmentation.get("augment_train", True),
        limit_per_class=single_spec.get("limit_per_class"),
        limit_per_class_val=single_spec.get("limit_per_class_val"),
        limit_per_class_test=single_spec.get("limit_per_class_test"),
        medical_variant=single_spec.get("medical_variant"),
        use_monai_train_aug=data_cfg.get("use_monai", False),
        train_augmentation_library=augmentation.get("library", "torchvision"),
        eval_augmentation_library=augmentation.get("eval_library"),
        albumentations_variant=augmentation.get("variant", "advanced"),
        distributed=False,
        rank=0,
        world_size=1,
        collate_fn=collate_fn,
    )

    return evaluate_model(
        model,
        test_loader,
        device,
        distributed=False,
        world_size=1,
    )


def build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    registry = model_cfg.get("registry")
    key = model_cfg.get("key")
    params = model_cfg.get("params", {})
    if registry is None or key is None:
        raise ValueError("Model configuration requires 'registry' and 'key'.")

    registry = registry.lower()
    if registry == "advanced_cnn":
        return AdvancedCNNRegistry.create(key, **params)
    if registry == "vision_transformer":
        return VisionTransformerRegistry.create(key, **params)
    if registry == "transfer_learning":
        return TransferLearningRegistry.get_model(key, **params)
    if registry == "foundation_general":
        return FoundationGeneralRegistry.create(key, **params)
    if registry == "foundation_medical":
        return FoundationMedicalRegistry.create(key, **params)
    if registry in {"basic_cnn", "cnn"}:
        return CNNRegistry.get_model(key, **params)

    raise ValueError(f"Unsupported model registry '{registry}'.")


def build_data(
    cfg: Dict[str, Any],
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    collate_fn: Optional[Callable] = None,
) -> Tuple[Any, Any, Any]:
    augmentation = cfg.get("augmentation", {})
    train_library = augmentation.get("library", "torchvision")
    if cfg.get("use_monai", False):
        train_library = "monai"
    multi_cfg = cfg.get("multi_source")
    if multi_cfg and multi_cfg.get("enabled", True):
        dataset_specs = multi_cfg.get("datasets") or []
        if not dataset_specs:
            roots = multi_cfg.get("roots") or []
            dataset_specs = [{"root": root} for root in roots]
        if not dataset_specs:
            raise ValueError(
                "Multi-source configuration requires at least one dataset root"
            )

        default_limit = cfg.get("limit_per_class")
        default_limit_val = cfg.get("limit_per_class_val")
        default_limit_test = cfg.get("limit_per_class_test")
        for spec in dataset_specs:
            spec.setdefault("limit_per_class", default_limit)
            spec.setdefault("limit_per_class_val", default_limit_val)
            spec.setdefault("limit_per_class_test", default_limit_test)
            spec.setdefault("medical_variant", cfg.get("medical_variant", "none"))

        return create_multi_source_data_loaders(
            dataset_specs,
            batch_size=cfg.get("batch_size", 32),
            target_size=_parse_target_size(cfg.get("target_size", _MISSING)),
            num_workers=cfg.get("num_workers", 4),
            augment_train=augmentation.get("augment_train", True),
            pin_memory=cfg.get("pin_memory"),
            default_medical_variant=cfg.get("medical_variant", "none"),
            train_augmentation_library=train_library,
            eval_augmentation_library=augmentation.get("eval_library"),
            albumentations_variant=augmentation.get("variant", "advanced"),
            sampling_strategy=multi_cfg.get("sampling", "proportional"),
            collate_fn=collate_fn,
        )

    loaders = create_data_loaders(
        data_root=cfg.get("root", "dataset/set_a"),
        batch_size=cfg.get("batch_size", 32),
        target_size=_parse_target_size(cfg.get("target_size", _MISSING)),
        num_workers=cfg.get("num_workers", 4),
        augment_train=augmentation.get("augment_train", True),
        limit_per_class=cfg.get("limit_per_class"),
        limit_per_class_val=cfg.get("limit_per_class_val"),
        limit_per_class_test=cfg.get("limit_per_class_test"),
        medical_variant=cfg.get("medical_variant", "none"),
        use_monai_train_aug=cfg.get("use_monai", False),
        train_augmentation_library=train_library,
        eval_augmentation_library=augmentation.get("eval_library"),
        albumentations_variant=augmentation.get("variant", "advanced"),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        collate_fn=collate_fn,
    )
    return loaders


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    distributed: bool = False,
    world_size: int = 1,
) -> Dict[str, float]:
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    num_classes = getattr(base_model, "num_classes", None)

    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(0)

    preds_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    probs_list: list[torch.Tensor] = []

    supports_sizes = _model_supports_original_sizes(model)

    with torch.no_grad():
        for batch in dataloader:
            images, labels, sizes = _unwrap_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sizes_list: Optional[List[Tuple[int, int]]] = None
            if supports_sizes and sizes is not None:
                sizes_list = [tuple(map(int, pair)) for pair in sizes.tolist()]
            if supports_sizes and sizes_list:
                outputs = model(images, original_sizes=sizes_list)
            else:
                outputs = model(images)
            logits, _ = extract_logits_from_output(outputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds_list.append(preds.detach())
            labels_list.append(labels.detach())
            if num_classes:
                probs_list.append(probs.detach())

    if preds_list:
        preds_tensor = torch.cat(preds_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
    else:
        preds_tensor = torch.empty(0, dtype=torch.long, device=device)
        labels_tensor = torch.empty(0, dtype=torch.long, device=device)
    probs_tensor: Optional[torch.Tensor]
    if probs_list:
        probs_tensor = torch.cat(probs_list, dim=0)
    else:
        probs_tensor = None

    if distributed and world_size > 1:
        preds_tensor = _gather_variable_length_tensor(preds_tensor, world_size)
        labels_tensor = _gather_variable_length_tensor(labels_tensor, world_size)
        if probs_tensor is not None and probs_tensor.numel() > 0:
            probs_tensor = _gather_variable_length_tensor(probs_tensor, world_size)

    preds_np = preds_tensor.cpu().numpy()
    labels_np = labels_tensor.cpu().numpy()
    probs_np = (
        probs_tensor.cpu().numpy()
        if probs_tensor is not None and probs_tensor.numel() > 0
        else None
    )
    return classification_metrics(
        labels_np, preds_np, probs=probs_np, num_classes=num_classes
    )


def append_summary(summary_path: Path, record: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def maybe_enable_distillation(
    trainer: DeepLearningTrainer,
    kd_cfg: Dict[str, Any] | None,
    device: torch.device,
) -> None:
    config = DistillationConfig.from_dict(kd_cfg)
    if not config.should_enable():
        return

    teacher = build_teacher_from_config(config, build_model, device)
    trainer.set_teacher(
        teacher,
        alpha=config.alpha,
        temperature=config.temperature,
    )


def maybe_setup_curriculum(
    trainer: DeepLearningTrainer, curriculum_cfg: Dict[str, Any] | None
) -> None:
    hook = build_curriculum_hook(curriculum_cfg)
    if hook is not None:
        trainer.set_curriculum_hook(hook)


def maybe_generate_attention(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
) -> None:
    eval_cfg = cfg.get("evaluation", {})
    attention_cfg = eval_cfg.get("attention_viz", {})
    if not attention_cfg.get("enable"):
        return

    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0,
        )

    visualizer = ResultVisualizer(
        model,
        device=device,
        class_names=getattr(dataloader.dataset, "class_names", None),
    )
    viz_config = AttentionVizConfig(
        samples_per_class=attention_cfg.get("samples_per_class", 2),
        rollout=attention_cfg.get("rollout", False),
        colormap=attention_cfg.get("colormap", "jet"),
        seed=attention_cfg.get("seed"),
    )
    target_dir = Path(attention_cfg.get("output_dir", output_dir / "attention"))
    visualizer.generate_attention_maps(dataloader, target_dir, viz_config)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Training and evaluation utilities for Stage 4 models."""


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--device", "device_str", type=str, default=None)
@click.option(
    "--summary-output",
    type=click.Path(path_type=Path),
    default=Path("experiments/results/advanced_dl_summary.json"),
)
@click.option("--quiet", is_flag=True, help="Disable tqdm progress bars.")
@click.option(
    "--skip-attention",
    is_flag=True,
    help="Skip attention visualization even if config enables it.",
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint (.pth) to resume training from.",
)
def train(
    config_path: Path,
    device_str: Optional[str],
    summary_output: Path,
    quiet: bool,
    skip_attention: bool,
    resume_from: Optional[Path],
) -> None:
    """Train a model using a configuration YAML file."""
    cfg = load_config(config_path)
    exp_cfg = cfg.get("experiment", {})
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    collate_fn = pad_image_collate if _requires_variable_image_collate(model_cfg) else None
    dist_setting = train_cfg.get("distributed", "auto")
    dist_flag = _normalize_bool(dist_setting)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = False

    if dist_flag is None:
        distributed = env_world_size > 1 and dist.is_available()
    else:
        distributed = bool(dist_flag and env_world_size > 1 and dist.is_available())

    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if distributed else None
    device = resolve_device(device_str, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if not _init_distributed(backend):
            distributed = False
            local_rank = None

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0

    if dist_flag and not distributed and is_main and env_world_size <= 1:
        click.echo(
            "⚠️ Distributed training requested but only a single process detected; falling back to single-GPU mode."
        )

    effective_quiet = quiet or (distributed and not is_main)

    if is_main:
        click.echo(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_data(
        cfg.get("data", {}),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        collate_fn=collate_fn,
    )

    model = build_model(deepcopy(model_cfg))
    # Quiet 模式时避免批次级日志泛滥：将 log_interval 设置得非常大
    log_interval = train_cfg.get("log_interval", 20)
    if effective_quiet:
        log_interval = max(log_interval, 10**9)

    trainer = DeepLearningTrainer(
        model,
        device=device,
        output_dir=Path(exp_cfg.get("output_dir", "experiments/models/advanced_dl")),
        use_tqdm=not effective_quiet,
        log_interval=log_interval,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main,
        ddp_find_unused_parameters=train_cfg.get(
            "ddp_find_unused_parameters", distributed
        ),
        model_selection_metric=train_cfg.get("model_selection_metric", "loss"),
    )
    single_device_eval_cfg = train_cfg.get("single_device_eval")
    single_device_eval = (
        bool(single_device_eval_cfg)
        if single_device_eval_cfg is not None
        else distributed
    )
    default_eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evaluation_device_str = train_cfg.get(
        "evaluation_device", default_eval_device
    )

    opt_cfg = cfg.get("optimization", {})
    sched_cfg = cfg.get("scheduler", {})
    class_weights_tensor = None
    if opt_cfg.get("class_weights") is not None:
        class_weights_tensor = torch.tensor(
            opt_cfg["class_weights"], dtype=torch.float32
        )

    # Parse criterion config (support both old string format and new dict format)
    criterion_config = opt_cfg.get("criterion", "cross_entropy")
    if isinstance(criterion_config, dict):
        # New format: criterion: {name: coral, importance_weights: [...]}
        criterion_name = criterion_config.get("name", "cross_entropy")
        criterion_params = criterion_config
    else:
        # Old format: criterion: "cross_entropy" (string)
        criterion_name = criterion_config
        criterion_params = opt_cfg

    trainer.setup_training(
        optimizer_name=opt_cfg.get("optimizer", "adamw"),
        learning_rate=opt_cfg.get("learning_rate", 1e-3),
        lr_backbone=opt_cfg.get("backbone_learning_rate"),
        lr_head=opt_cfg.get("head_learning_rate"),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        scheduler_name=sched_cfg.get("name"),
        scheduler_params=sched_cfg.get("params"),
        criterion_name=criterion_name,
        criterion_params=criterion_params,
        class_weights=class_weights_tensor,
        early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        grad_clip_norm=opt_cfg.get("grad_clip_norm"),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
        use_amp=train_cfg.get("amp", False),
        amp_dtype=train_cfg.get("amp_dtype", "float16"),
    )

    maybe_enable_distillation(trainer, cfg.get("knowledge_distillation"), device)
    maybe_setup_curriculum(trainer, cfg.get("curriculum"))

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=train_cfg.get("epochs", 30),
        save_best=train_cfg.get("save_best", True),
        save_last=train_cfg.get("save_last", True),
        resume_from=resume_from,
        verbose=is_main and not effective_quiet,
    )

    if distributed:
        dist.barrier()

    summary = trainer.metrics.get_summary() if is_main else {}
    best_epoch = summary.get("best_epoch", 0) + 1
    if distributed:
        epoch_tensor = torch.tensor([best_epoch], device=device, dtype=torch.long)
        dist.broadcast(epoch_tensor, src=0)
        best_epoch = int(epoch_tensor.item())

    best_path = trainer.output_dir / f"best_model_epoch_{best_epoch}.pth"
    if train_cfg.get("save_best", True) and best_path.exists():
        trainer.load_model(best_path)

    test_metrics: Dict[str, float] = {}
    model_for_attention = trainer.model
    loader_for_attention = test_loader
    attention_device = device
    if distributed and single_device_eval:
        dist.barrier()
        _cleanup_distributed()
        distributed = False
        world_size = 1
        if not is_main:
            return
        eval_device = resolve_device(evaluation_device_str)
        click.echo(f"🧪 Evaluation running on single device: {eval_device}")
        eval_model = build_model(deepcopy(model_cfg))
        eval_trainer = DeepLearningTrainer(
            eval_model,
            device=eval_device,
            output_dir=trainer.output_dir,
            use_tqdm=False,
        )
        eval_trainer.load_model(best_path)
        _, _, eval_test_loader = build_data(
            cfg.get("data", {}),
            distributed=False,
            collate_fn=collate_fn,
        )
        test_metrics = evaluate_model(
            eval_trainer.model,
            eval_test_loader,
            eval_device,
            distributed=False,
            world_size=1,
        )
        model_for_attention = eval_trainer.model
        loader_for_attention = eval_test_loader
        attention_device = eval_device
    else:
        if distributed:
            dist.barrier()
        test_metrics = evaluate_model(
            trainer.model,
            test_loader,
            device,
            distributed=distributed,
            world_size=world_size,
        )

    per_source_test_metrics: Dict[str, Dict[str, float]] = {}
    if is_main:
        data_cfg_top = cfg.get("data", {})
        multi_cfg = data_cfg_top.get("multi_source", {})
        eval_cfg = cfg.get("evaluation", {})
        eval_per_source_flag = eval_cfg.get("per_source_eval")
        should_eval_per_source = (
            multi_cfg
            and multi_cfg.get("enabled", True)
            and (eval_per_source_flag is None or bool(eval_per_source_flag))
        )
        if should_eval_per_source:
            dataset_specs = []
            if isinstance(multi_cfg.get("datasets"), list):
                dataset_specs = [
                    spec
                    for spec in multi_cfg["datasets"]
                    if isinstance(spec, dict) and spec.get("root")
                ]
            elif isinstance(multi_cfg.get("roots"), list):
                dataset_specs = [
                    {"root": root}
                    for root in multi_cfg["roots"]
                    if isinstance(root, (str, Path)) and root
                ]
            if dataset_specs:
                for spec in dataset_specs:
                    label = spec.get("name") or _normalize_dataset_name(spec.get("root"))
                    metrics = _evaluate_single_dataset(
                        cfg,
                        spec,
                        model_for_attention,
                        attention_device,
                        collate_fn=collate_fn,
                    )
                    per_source_test_metrics[label] = metrics

    if is_main:
        record = {
            "experiment_name": exp_cfg.get("name", config_path.stem),
            "config": str(config_path),
            "output_dir": str(trainer.output_dir),
            "figures_dir": exp_cfg.get("figures_dir"),
            "reports_dir": exp_cfg.get("reports_dir"),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_acc": summary.get("best_val_acc"),
            "test_metrics": test_metrics,
            "per_source_test_metrics": per_source_test_metrics or None,
            "history": history,
        }

        append_summary(summary_output, record)

        metadata = {
            "config": str(config_path),
            "experiment": exp_cfg,
            "model": cfg.get("model", {}),
            "data": cfg.get("data", {}),
            "optimization": cfg.get("optimization", {}),
            "scheduler": cfg.get("scheduler", {}),
            "training": cfg.get("training", {}),
            "knowledge_distillation": cfg.get("knowledge_distillation", {}),
            "results": {
                "summary": summary,
                "test_metrics": test_metrics,
                "per_source_test_metrics": per_source_test_metrics or None,
                "best_model_path": str(best_path),
            },
        }
        with open(trainer.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        figures_root = Path(exp_cfg.get("figures_dir", trainer.output_dir / "figures"))
        if not skip_attention:
            maybe_generate_attention(
                cfg,
                model_for_attention,
                loader_for_attention,
                attention_device,
                figures_root,
            )

        click.echo(
            json.dumps(
                {
                    "best_val_acc": summary.get("best_val_acc"),
                    "test_metrics": test_metrics,
                    "per_source_test_metrics": per_source_test_metrics or None,
                },
                indent=2,
            )
        )

        # Auto-prune extra checkpoints: keep best and last by default.
        # Set KEEP_ALL_CHECKPOINTS=1 to skip pruning; set DROP_LAST_CHECKPOINT=1 to drop last_model.pth.
        keep_all = os.environ.get("KEEP_ALL_CHECKPOINTS") in {"1", "true", "True"}
        drop_last = os.environ.get("DROP_LAST_CHECKPOINT") in {"1", "true", "True"}
        if not keep_all:
            prune_checkpoints(trainer.output_dir, best_path, keep_last=not drop_last)

    if distributed:
        _cleanup_distributed()


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--device", "device_str", type=str, default=None)
@click.option("--summary-output", type=click.Path(path_type=Path), default=None)
def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    device_str: Optional[str],
    summary_output: Optional[Path],
) -> None:
    """Evaluate a checkpoint on the dataset specified by the config."""
    cfg = load_config(config_path)
    device = resolve_device(device_str)
    click.echo(f"Using device: {device}")

    model_cfg = cfg.get("model", {})
    collate_fn = pad_image_collate if _requires_variable_image_collate(model_cfg) else None

    _, _, test_loader = build_data(
        cfg.get("data", {}),
        distributed=False,
        collate_fn=collate_fn,
    )
    model = build_model(deepcopy(model_cfg))

    trainer = DeepLearningTrainer(
        model, device=device, output_dir=Path("/tmp"), use_tqdm=False
    )
    trainer.load_model(checkpoint_path)

    metrics = evaluate_model(trainer.model, test_loader, device)
    click.echo(json.dumps(metrics, indent=2))

    if summary_output:
        record = {
            "experiment_name": cfg.get("experiment", {}).get("name", config_path.stem),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "test_metrics": metrics,
        }
        append_summary(summary_output, record)


if __name__ == "__main__":
    main()
