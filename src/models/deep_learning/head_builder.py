"""Shared helpers for constructing classifier heads."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from src.models.heads.ordinal import OrdinalRegressionHead


def build_classification_head(
    hidden_size: int,
    num_classes: int,
    *,
    head_cfg: Optional[Dict[str, Any]] = None,
    default_dropout: float = 0.0,
) -> nn.Module:
    """Factory for classifier heads with optional ordinal regression support."""

    cfg = dict(head_cfg or {})
    head_type = str(cfg.get("type", "linear")).lower()
    dropout = float(cfg.get("dropout", default_dropout or 0.0))

    if head_type == "ordinal":
        mode = cfg.get("mode", "coral")
        return OrdinalRegressionHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            mode=mode,
            dropout=dropout,
        )

    if head_type == "mlp":
        hidden_dim = int(cfg.get("hidden_dim", hidden_size))
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_size, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes)])
        return nn.Sequential(*layers)

    # Default linear head with optional dropout
    if dropout > 0.0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
    return nn.Linear(hidden_size, num_classes)
