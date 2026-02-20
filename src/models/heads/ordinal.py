"""Ordinal regression head utilities (CORAL/CORN) for KL grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_levels(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class labels to ordinal levels for CORAL/CORN losses."""

    if targets.dtype not in (torch.int32, torch.int64):
        targets = targets.long()
    targets = targets.view(-1, 1)
    device = targets.device
    arange = torch.arange(num_classes - 1, device=device).view(1, -1)
    levels = (targets > arange).float()
    return levels


def coral_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL logits to class probability distribution."""

    prob_gt = torch.sigmoid(logits)  # P(y > k)
    first = 1.0 - prob_gt[:, :1]
    middle = prob_gt[:, :-1] - prob_gt[:, 1:] if prob_gt.size(1) > 1 else torch.empty(0, device=logits.device)
    last = prob_gt[:, -1:]
    probs = torch.cat([first, middle, last], dim=1) if middle.numel() else torch.cat([first, last], dim=1)
    probs = probs.clamp_(min=1e-6, max=1.0)
    return probs / probs.sum(dim=1, keepdim=True)


def corn_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits to class probability distribution."""

    prob_gt = torch.sigmoid(logits)
    batch, k_minus_1 = prob_gt.shape
    probs = []
    running = torch.ones(batch, device=logits.device)
    for idx in range(k_minus_1):
        prob = running * (1.0 - prob_gt[:, idx])
        probs.append(prob.unsqueeze(1))
        running = running * prob_gt[:, idx]
    probs.append(running.unsqueeze(1))
    probs_tensor = torch.cat(probs, dim=1).clamp_(min=1e-6, max=1.0)
    return probs_tensor / probs_tensor.sum(dim=1, keepdim=True)


def ordinal_probs_from_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """Utility that dispatches to CORAL or CORN probability conversion."""

    mode = mode.lower()
    if mode == "coral":
        return coral_probs_from_logits(logits)
    if mode == "corn":
        return corn_probs_from_logits(logits)
    raise ValueError(f"Unsupported ordinal mode '{mode}'.")


@dataclass
class OrdinalHeadOutput:
    """Wrapper so downstream code can access logits like Hugging Face outputs."""

    logits: torch.Tensor
    ordinal_logits: torch.Tensor
    probs: torch.Tensor
    mode: str


class OrdinalRegressionHead(nn.Module):
    """Projection head that produces ordinal logits for CORAL/CORN losses."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        mode: str = "coral",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Ordinal head requires at least 2 classes.")
        mode = mode.lower()
        if mode not in {"coral", "corn"}:
            raise ValueError(f"Unsupported ordinal mode '{mode}'. Expected 'coral' or 'corn'.")
        self.num_classes = num_classes
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.linear = nn.Linear(hidden_size, num_classes - 1)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> OrdinalHeadOutput:
        if self.dropout is not None:
            features = self.dropout(features)
        ordinal_logits = self.linear(features)
        probs = ordinal_probs_from_logits(ordinal_logits, self.mode)
        logits = torch.log(probs)
        return OrdinalHeadOutput(logits=logits, ordinal_logits=ordinal_logits, probs=probs, mode=self.mode)


def extract_logits_from_output(output: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (class_logits, ordinal_logits) tuple from arbitrary model outputs."""

    ordinal_logits = getattr(output, "ordinal_logits", None)
    if isinstance(output, OrdinalHeadOutput):
        return output.logits, ordinal_logits
    if isinstance(output, dict):
        logits = output.get("logits")
        ordinal_logits = output.get("ordinal_logits", ordinal_logits)
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output
    if logits is None:
        raise ValueError("Model output did not contain logits.")
    return logits, ordinal_logits
