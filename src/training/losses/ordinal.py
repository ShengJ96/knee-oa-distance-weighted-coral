"""Ordinal regression losses used by Stage 6 (CORAL / CORN)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads.ordinal import label_to_levels


class CORALLoss(nn.Module):
    """Implements the CORAL (Cumulative Ordinal Regression) loss.

    Args:
        num_classes: Number of ordinal classes
        reduction: Loss reduction method ('mean', 'sum', or 'none')
        importance_weights: Optional tensor of shape (num_classes-1,) with weights for each threshold.
                          If None, all thresholds are weighted equally.

    Example:
        # For 5 classes with imbalanced thresholds
        weights = torch.tensor([0.5, 2.0, 3.0, 5.0])  # Higher weight for rare thresholds
        loss_fn = CORALLoss(num_classes=5, importance_weights=weights)
    """

    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean",
        importance_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL loss requires at least two classes.")
        self.num_classes = num_classes
        self.reduction = reduction

        # Register importance_weights as buffer (moved to device automatically)
        if importance_weights is not None:
            if importance_weights.shape != (num_classes - 1,):
                raise ValueError(
                    f"importance_weights must have shape ({num_classes - 1},), "
                    f"got {importance_weights.shape}"
                )
            self.register_buffer("importance_weights", importance_weights)
        else:
            self.register_buffer("importance_weights", None, persistent=False)

    def forward(self, ordinal_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        levels = label_to_levels(targets, self.num_classes)

        # Compute per-threshold BCE loss
        per_threshold_loss = F.binary_cross_entropy_with_logits(
            ordinal_logits, levels, reduction="none"  # type: ignore[arg-type]
        )  # shape: (batch_size, num_classes-1)

        # Apply importance weights if provided
        if self.importance_weights is not None:
            per_threshold_loss = per_threshold_loss * self.importance_weights

        # Apply reduction
        if self.reduction == "mean":
            return per_threshold_loss.mean()
        elif self.reduction == "sum":
            return per_threshold_loss.sum()
        else:  # 'none'
            return per_threshold_loss


class CostSensitiveCORALLoss(nn.Module):
    """Cost-Sensitive CORAL Loss with clinical cost matrix.

    This loss extends CORAL by incorporating a cost matrix C[i,j] that represents
    the clinical cost of misclassifying true class i as predicted class j.
    For ordinal classification (e.g., KL 0-4), this allows encoding:
    - Ordinal nature: Adjacent misclassifications (KL 2→3) have lower cost than
      large jumps (KL 0→4)
    - Clinical significance: Misclassifying severe cases as normal is more costly
      than the reverse
    - Class imbalance: Automatically weights rare classes higher

    Args:
        num_classes: Number of ordinal classes (e.g., 5 for KL 0-4)
        cost_matrix: Either a string specifying the cost type or a tensor:
            - 'linear': C[i,j] = |i - j| (MAE-aligned)
            - 'quadratic': C[i,j] = (i - j)^2 (QWK-aligned, default)
            - 'clinical': Quadratic with KL≥2 threshold penalty
            - Tensor of shape (num_classes, num_classes): Custom cost matrix
        reduction: 'mean', 'sum', or 'none'
        clinical_threshold: For 'clinical' mode, the threshold index (default=2 for KL≥2)
        clinical_penalty: For 'clinical' mode, multiplier for cross-threshold errors (default=1.5)

    Example:
        >>> # Quadratic cost for KL grading
        >>> loss_fn = CostSensitiveCORALLoss(num_classes=5, cost_matrix='quadratic')
        >>>
        >>> # Clinical cost with KL≥2 threshold emphasis
        >>> loss_fn = CostSensitiveCORALLoss(
        ...     num_classes=5,
        ...     cost_matrix='clinical',
        ...     clinical_threshold=2,
        ...     clinical_penalty=1.5
        ... )
    """

    def __init__(
        self,
        num_classes: int,
        cost_matrix: str | torch.Tensor = "quadratic",
        reduction: str = "mean",
        clinical_threshold: int = 2,
        clinical_penalty: float = 1.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Cost-sensitive CORAL requires at least two classes.")

        self.num_classes = num_classes
        self.reduction = reduction

        # Build or validate cost matrix
        if isinstance(cost_matrix, str):
            C = self._build_cost_matrix(
                cost_matrix, num_classes, clinical_threshold, clinical_penalty
            )
        else:
            C = cost_matrix
            if C.shape != (num_classes, num_classes):
                raise ValueError(
                    f"Cost matrix must have shape ({num_classes}, {num_classes}), "
                    f"got {C.shape}"
                )

        # Convert cost matrix to threshold-level weights
        threshold_weights = self._cost_matrix_to_threshold_weights(C, num_classes)
        self.register_buffer("threshold_weights", threshold_weights)
        self.register_buffer("cost_matrix", C)

    @staticmethod
    def _build_cost_matrix(
        cost_type: str,
        num_classes: int,
        clinical_threshold: int = 2,
        clinical_penalty: float = 1.5,
    ) -> torch.Tensor:
        """Build predefined cost matrices for ordinal classification.

        Args:
            cost_type: 'linear', 'quadratic', or 'clinical'
            num_classes: Number of classes
            clinical_threshold: Threshold for clinical weighting (e.g., KL≥2)
            clinical_penalty: Penalty multiplier for cross-threshold errors

        Returns:
            Cost matrix of shape (num_classes, num_classes)
        """
        indices = torch.arange(num_classes, dtype=torch.float32)
        i, j = torch.meshgrid(indices, indices, indexing='ij')

        if cost_type == "linear":
            # C[i,j] = |i - j|
            C = torch.abs(i - j)
        elif cost_type == "quadratic":
            # C[i,j] = (i - j)^2
            C = (i - j) ** 2
        elif cost_type == "clinical":
            # Quadratic base + penalty for cross-threshold errors
            C = (i - j) ** 2
            # Penalty when misclassification crosses the clinical threshold
            # E.g., predicting KL 0-1 when true is 2-4, or vice versa
            cross_threshold = (i < clinical_threshold) != (j < clinical_threshold)
            C = torch.where(cross_threshold, C * clinical_penalty, C)
        else:
            raise ValueError(
                f"Unknown cost_type: {cost_type}. "
                f"Must be 'linear', 'quadratic', or 'clinical'."
            )

        return C

    @staticmethod
    def _cost_matrix_to_threshold_weights(
        cost_matrix: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Convert a (num_classes, num_classes) cost matrix to (num_classes-1) threshold weights.

        For each threshold k in [0, num_classes-2]:
        - Threshold k separates classes [0..k] from [k+1..num_classes-1]
        - We compute the average cost of all misclassifications that involve this threshold:
          * False Positive (k): true class <= k, predicted class > k
          * False Negative (k): true class > k, predicted class <= k

        Args:
            cost_matrix: Cost matrix C[i,j] of shape (num_classes, num_classes)
            num_classes: Number of classes

        Returns:
            Tensor of shape (num_classes-1,) with weights for each threshold
        """
        weights = []
        for k in range(num_classes - 1):
            # Classes below and including threshold k: [0..k]
            # Classes above threshold k: [k+1..num_classes-1]

            # False Positive: true in [0..k], predicted in [k+1..num_classes-1]
            fp_costs = cost_matrix[0:k+1, k+1:num_classes]

            # False Negative: true in [k+1..num_classes-1], predicted in [0..k]
            fn_costs = cost_matrix[k+1:num_classes, 0:k+1]

            # Average cost of all misclassifications involving this threshold
            all_costs = torch.cat([fp_costs.flatten(), fn_costs.flatten()])
            avg_cost = all_costs.mean()
            weights.append(avg_cost)

        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, ordinal_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cost-sensitive CORAL loss.

        Args:
            ordinal_logits: Logits of shape (batch_size, num_classes-1)
            targets: Integer labels of shape (batch_size,) in range [0, num_classes-1]

        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
        """
        levels = label_to_levels(targets, self.num_classes)

        # Compute per-threshold BCE loss
        per_threshold_loss = F.binary_cross_entropy_with_logits(
            ordinal_logits, levels, reduction="none"  # type: ignore[arg-type]
        )  # shape: (batch_size, num_classes-1)

        # Apply cost-based threshold weights
        weighted_loss = per_threshold_loss * self.threshold_weights

        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss


class CORNLoss(nn.Module):
    """Conditional Ordinal Regression for Neural networks (CORN) loss."""

    def __init__(self, num_classes: int, reduction: str = "mean") -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORN loss requires at least two classes.")
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, ordinal_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        total = 0
        for k in range(self.num_classes - 1):
            mask = targets >= k
            if not torch.any(mask):
                continue
            binary_targets = (targets[mask] > k).float()
            logits = ordinal_logits[mask, k]
            loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="mean")
            losses.append(loss)
            total += 1
        if not losses:
            return torch.tensor(0.0, device=ordinal_logits.device, requires_grad=ordinal_logits.requires_grad)
        stacked = torch.stack(losses)
        return stacked.mean() if self.reduction == "mean" else stacked.sum()
