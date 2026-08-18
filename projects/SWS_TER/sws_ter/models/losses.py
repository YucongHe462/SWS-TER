"""Loss primitives used by SWS-TER."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def region_prior_weight(target_prior: Tensor,
                        background_prior: Tensor,
                        hard_background_prior: Tensor,
                        base_weight: float = 1.0,
                        lambda_target: float = 1.05,
                        lambda_background: float = 0.95,
                        lambda_hard: float = 0.85,
                        minimum: float = 0.05,
                        maximum: float = 2.0) -> Tensor:
    """Equation (32): ACPC-modulated negative-sample weight."""
    weight = (base_weight + lambda_background * background_prior
              + lambda_hard * hard_background_prior
              - lambda_target * target_prior)
    return weight.clamp(minimum, maximum)


def sparse_prior_focal_loss(logits: Tensor,
                            targets: Tensor,
                            prior_weight: Optional[Tensor] = None,
                            alpha: float = 0.25,
                            gamma: float = 2.0,
                            hard_negative_threshold: float = 0.5,
                            hard_negative_modulation: float = 0.4,
                            reduction: str = 'mean') -> Tensor:
    """Equation (33) in numerically stable logit form.

    ``targets`` is a binary tensor with the same shape as ``logits``.  ACPC
    weights affect negative entries only; positives retain the ordinary focal
    term.  The manuscript does not fix ``thr``, so it remains configurable.
    """
    targets = targets.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    positive = targets > 0.5
    focal = torch.where(positive, (1 - probability).pow(gamma),
                        probability.pow(gamma))
    balance = torch.where(positive,
                          logits.new_tensor(alpha),
                          logits.new_tensor(1 - alpha))
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none') * focal * balance
    if prior_weight is None:
        prior_weight = torch.ones_like(loss)
    while prior_weight.ndim < loss.ndim:
        prior_weight = prior_weight.unsqueeze(-1)
    hard_negative = (~positive) & (probability > hard_negative_threshold)
    negative_weight = prior_weight * hard_negative_modulation
    loss = torch.where(hard_negative, loss * negative_weight, loss)
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    if reduction != 'mean':
        raise ValueError(f'unsupported reduction: {reduction}')
    return loss.mean()


def quality_focal_distillation(student_logits: Tensor,
                               teacher_probabilities: Tensor,
                               weights: Tensor,
                               beta: float = 2.0) -> Tensor:
    """Dense classification distillation used on high-confidence candidates."""
    student_probability = student_logits.sigmoid()
    zero_target = torch.zeros_like(student_probability)
    loss = F.binary_cross_entropy_with_logits(
        student_logits, zero_target, reduction='none') * student_probability.pow(beta)
    positive = weights > 0
    if positive.any():
        difference = (teacher_probabilities[positive]
                      - student_probability[positive]).abs().pow(beta)
        loss[positive] = F.binary_cross_entropy_with_logits(
            student_logits[positive], teacher_probabilities[positive],
            reduction='none') * difference
    expanded = weights
    while expanded.ndim < loss.ndim:
        expanded = expanded.unsqueeze(-1)
    denominator = expanded.sum().clamp_min(1e-6)
    return (loss * expanded).sum() / denominator

