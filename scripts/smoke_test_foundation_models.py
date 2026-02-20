"""Quick smoke tests for Stage 5 foundation model registries.

Usage examples:
  uv run python scripts/smoke_test_foundation_models.py
  uv run python scripts/smoke_test_foundation_models.py --models foundation_general:siglip_base_patch16_384 foundation_medical:radimagenet_resnet50
  uv run python scripts/smoke_test_foundation_models.py --pretrained --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from src.models.deep_learning.foundation_general import FoundationGeneralRegistry
from src.models.deep_learning.foundation_medical import FoundationMedicalRegistry


@dataclass
class ModelSpec:
    registry: str
    key: str
    input_size: Tuple[int, int]


DEFAULT_MODELS: Tuple[Tuple[str, str], ...] = (
    ("foundation_general", "siglip_base_patch16_384"),
    ("foundation_general", "siglip2_base_patch16_384"),
    ("foundation_general", "siglip2_so400m_patch14_384"),
    ("foundation_general", "dinov2_vit_l14"),
    ("foundation_medical", "radimagenet_resnet50"),
)


def resolve_spec(registry: str, key: str) -> ModelSpec:
    if registry == "foundation_general":
        spec = FoundationGeneralRegistry.get_spec(key)
    elif registry == "foundation_medical":
        spec = FoundationMedicalRegistry.get_spec(key)
    else:
        raise ValueError(f"Unsupported registry '{registry}'.")
    return ModelSpec(registry=registry, key=key, input_size=spec.input_size)


def run_smoke(spec: ModelSpec, *, batch_size: int, num_classes: int, pretrained: bool, device: torch.device) -> None:
    h, w = spec.input_size
    inputs = torch.randn(batch_size, 3, h, w, device=device)

    kwargs = {"num_classes": num_classes, "pretrained": pretrained}
    if spec.registry == "foundation_general":
        model = FoundationGeneralRegistry.create(spec.key, **kwargs)
    else:
        model = FoundationMedicalRegistry.create(spec.key, **kwargs)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    print(f"✓ {spec.registry}:{spec.key} | input {batch_size}x3x{h}x{w} → logits {tuple(outputs.shape)}")


def parse_model_list(entries: Iterable[str]) -> list[Tuple[str, str]]:
    parsed = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Model '{entry}' must be of the form registry:key")
        registry, key = entry.split(":", 1)
        parsed.append((registry.strip(), key.strip()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal forward passes on foundation registries")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to test in 'registry:key' format. Defaults to a small representative subset.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained weights (may download large files)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = parse_model_list(args.models) if args.models else list(DEFAULT_MODELS)

    device = torch.device(args.device)
    print(f"Running smoke tests on device: {device}")
    for registry, key in models:
        try:
            spec = resolve_spec(registry, key)
            run_smoke(
                spec,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {registry}:{key} failed — {exc}")


if __name__ == "__main__":
    main()
