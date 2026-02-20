"""Focal loss implementation for multi-class classification."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.alpha: Optional[torch.Tensor] = None
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                alpha_factor = self.alpha.to(logits.device)
            else:
                if self.alpha.numel() != num_classes:
                    raise ValueError(
                        "alpha length must match num_classes when provided as a sequence"
                    )
                alpha_factor = self.alpha.to(logits.device)[targets]
            loss = alpha_factor * focal_factor * ce
        else:
            loss = focal_factor * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
