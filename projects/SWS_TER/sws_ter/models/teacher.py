"""Uncertainty-guided supervision recovery teacher (Eqs. 34--41)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TorchGMM2:
    """Deterministic, device-local two-component 1-D Gaussian mixture.

    This avoids transferring dense predictions to scikit-learn/CPU at every
    iteration.  The returned probability always refers to the component with
    the larger fitted mean, as required by Eq. (36).
    """

    def __init__(self, iterations: int = 20, variance_floor: float = 1e-5,
                 tolerance: float = 1e-5) -> None:
        self.iterations = int(iterations)
        self.variance_floor = float(variance_floor)
        self.tolerance = float(tolerance)

    @torch.no_grad()
    def posterior_high(self, scores: Tensor) -> Tensor:
        original_shape = scores.shape
        values = scores.detach().float().reshape(-1)
        if values.numel() == 0:
            return values.reshape(original_shape)
        if values.numel() < 4 or float(values.std(unbiased=False)) < 1e-6:
            minimum, maximum = values.min(), values.max()
            posterior = (values - minimum) / (maximum - minimum).clamp_min(1e-6)
            return posterior.reshape(original_shape).to(scores.dtype)
        means = torch.stack((torch.quantile(values, 0.25),
                             torch.quantile(values, 0.75)))
        variance = values.var(unbiased=False).clamp_min(self.variance_floor)
        variances = torch.stack((variance, variance))
        mixture = values.new_full((2,), 0.5)
        previous = means.clone()
        for _ in range(self.iterations):
            log_probability = (-0.5 * (
                torch.log(2 * torch.pi * variances)[None]
                + (values[:, None] - means[None]).square()
                / variances[None]) + mixture.clamp_min(1e-8).log()[None])
            responsibility = torch.softmax(log_probability, dim=1)
            mass = responsibility.sum(dim=0).clamp_min(1e-6)
            mixture = mass / values.numel()
            means = (responsibility * values[:, None]).sum(dim=0) / mass
            variances = (responsibility
                         * (values[:, None] - means[None]).square()).sum(dim=0) / mass
            variances.clamp_(min=self.variance_floor)
            if torch.max(torch.abs(means - previous)) < self.tolerance:
                break
            previous = means.clone()
        high_component = means.argmax()
        return responsibility[:, high_component].reshape(original_shape).to(scores.dtype)


class PrototypeGuidedReconstructor(nn.Module):
    """Lightweight ViT reconstructor operating on latent FPN features."""

    def __init__(self,
                 feature_dim: int = 256,
                 num_classes: int = 1,
                 embed_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.0,
                 prototype_momentum: float = 0.99,
                 max_context: int = 256,
                 max_uncertain: int = 128) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.num_classes = int(num_classes)
        self.prototype_momentum = float(prototype_momentum)
        self.max_context = int(max_context)
        self.max_uncertain = int(max_uncertain)
        self.input_projection = (nn.Identity() if feature_dim == embed_dim
                                 else nn.Linear(feature_dim, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.position_projection = nn.Sequential(
            nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.reconstructor = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.register_buffer('prototypes', torch.zeros(num_classes, embed_dim))
        self.register_buffer('prototype_initialized',
                             torch.zeros(num_classes, dtype=torch.bool))

    @staticmethod
    def _limited(indices: Tensor, scores: Tensor, limit: int) -> Tensor:
        if indices.numel() <= limit:
            return indices
        return indices[scores[indices].topk(limit).indices]

    @torch.no_grad()
    def update_prototypes(self, latent: Tensor, labels: Tensor) -> None:
        if latent.numel() == 0:
            return
        for class_id in labels.unique():
            index = int(class_id.item())
            candidate = latent[labels == class_id].mean(dim=0)
            if self.prototype_initialized[index]:
                self.prototypes[index].mul_(self.prototype_momentum).add_(
                    candidate, alpha=1.0 - self.prototype_momentum)
            else:
                self.prototypes[index].copy_(candidate)
                self.prototype_initialized[index] = True

    def forward(self,
                teacher_features: Tensor,
                teacher_logits: Tensor,
                student_logits: Tensor,
                high_mask: Tensor,
                uncertain_mask: Tensor,
                coordinates: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        zero = student_logits.sum() * 0.0
        teacher_probability = teacher_logits.detach().sigmoid()
        confidence, labels = teacher_probability.max(dim=1)
        high_indices = self._limited(high_mask.nonzero(as_tuple=False).flatten(),
                                     confidence, self.max_context)
        uncertain_indices = self._limited(
            uncertain_mask.nonzero(as_tuple=False).flatten(), confidence,
            self.max_uncertain)
        projected = self.input_projection(teacher_features.detach())
        self.update_prototypes(projected[high_indices], labels[high_indices])
        if uncertain_indices.numel() == 0:
            return {
                'loss_reconstruction': zero,
                'loss_distillation': zero,
                'uncertain_indices': uncertain_indices,
                'soft_labels': student_logits.new_zeros((0, self.num_classes)),
            }

        high_tokens = projected[high_indices]
        masked_tokens = self.mask_token.expand(uncertain_indices.numel(), -1)
        if coordinates is not None:
            normalized = coordinates.float()
            denominator = normalized.amax(dim=0, keepdim=True).clamp_min(1.0)
            positional = self.position_projection(normalized / denominator)
            high_tokens = high_tokens + positional[high_indices]
            masked_tokens = masked_tokens + positional[uncertain_indices]
        prototype_tokens = self.prototypes[self.prototype_initialized].detach()
        tokens = torch.cat((high_tokens, masked_tokens, prototype_tokens), dim=0)
        reconstructed = self.norm(self.reconstructor(tokens.unsqueeze(0))[0])
        start = high_tokens.shape[0]
        uncertain_features = reconstructed[start:start + masked_tokens.shape[0]]
        reconstructed_logits = self.classifier(uncertain_features)
        soft_labels = reconstructed_logits.sigmoid()
        target_labels = labels[uncertain_indices]
        ready = self.prototype_initialized[target_labels]
        if ready.any():
            target_prototype = self.prototypes[target_labels[ready]].detach()
            loss_reconstruction = (1 - F.cosine_similarity(
                uncertain_features[ready], target_prototype, dim=1)).mean()
        else:
            loss_reconstruction = zero
        loss_distillation = F.binary_cross_entropy_with_logits(
            student_logits[uncertain_indices], soft_labels.detach())
        return {
            'loss_reconstruction': loss_reconstruction,
            'loss_distillation': loss_distillation,
            'uncertain_indices': uncertain_indices,
            'soft_labels': soft_labels.detach(),
        }


@dataclass
class CandidatePartition:
    high: Tensor
    uncertain: Tensor
    low: Tensor
    posterior: Tensor
    joint_confidence: Tensor
    consistency: Tensor


class UncertaintyGuidedRecovery(nn.Module):
    """Level-specific GMM partition plus prototype reconstruction."""

    def __init__(self,
                 feature_dim: int = 256,
                 num_classes: int = 1,
                 high_posterior: float = 0.7,
                 low_posterior: float = 0.2,
                 high_confidence: float = 0.2,
                 uncertain_confidence: float = 0.02,
                 high_consistency: float = 0.5,
                 uncertain_consistency: float = 0.2,
                 reconstruction_weight: float = 0.5,
                 distillation_weight: float = 0.5,
                 reconstructor: Optional[dict] = None) -> None:
        super().__init__()
        self.high_posterior = float(high_posterior)
        self.low_posterior = float(low_posterior)
        self.high_confidence = float(high_confidence)
        self.uncertain_confidence = float(uncertain_confidence)
        self.high_consistency = float(high_consistency)
        self.uncertain_consistency = float(uncertain_consistency)
        self.reconstruction_weight = float(reconstruction_weight)
        self.distillation_weight = float(distillation_weight)
        self.gmm = TorchGMM2()
        self.reconstructor = PrototypeGuidedReconstructor(
            feature_dim=feature_dim, num_classes=num_classes,
            **dict(reconstructor or {}))

    @torch.no_grad()
    def partition(self,
                  teacher_logits: Tensor,
                  teacher_centerness: Tensor,
                  student_logits: Tensor,
                  level_slices: Sequence[slice],
                  student_centerness: Optional[Tensor] = None
                  ) -> CandidatePartition:
        teacher_probability = teacher_logits.sigmoid()
        class_confidence, class_index = teacher_probability.max(dim=1)
        center_probability = teacher_centerness.sigmoid().reshape(-1)
        joint = class_confidence * center_probability
        posterior = torch.zeros_like(joint)
        for level_slice in level_slices:
            posterior[level_slice] = self.gmm.posterior_high(joint[level_slice])
        student_probability = student_logits.detach().sigmoid()
        _, student_index = student_probability.max(dim=1)
        student_on_teacher_class = student_probability.gather(
            1, class_index[:, None]).reshape(-1)
        if student_centerness is None:
            student_joint = student_on_teacher_class
        else:
            student_joint = (student_on_teacher_class
                             * student_centerness.detach().sigmoid().reshape(-1))
        # Teacher-student consistency combines pseudo-category agreement with
        # proximity between the two branches' joint confidence scores.
        class_agreement = (student_index == class_index).to(joint.dtype)
        consistency = (1.0 - (joint - student_joint).abs()).clamp(0, 1)
        consistency = consistency * class_agreement
        high = ((posterior >= self.high_posterior)
                & (joint >= self.high_confidence)
                & (consistency >= self.high_consistency))
        low = ((posterior < self.low_posterior)
               | (joint < self.uncertain_confidence))
        uncertain = (~high & ~low
                     & (consistency >= self.uncertain_consistency))
        low = ~(high | uncertain)
        return CandidatePartition(
            high, uncertain, low, posterior, joint, consistency)

    def forward(self,
                teacher_features: Tensor,
                teacher_logits: Tensor,
                teacher_centerness: Tensor,
                student_logits: Tensor,
                level_slices: Sequence[slice],
                coordinates: Optional[Tensor] = None,
                student_centerness: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        partition = self.partition(teacher_logits, teacher_centerness,
                                   student_logits, level_slices,
                                   student_centerness=student_centerness)
        output = self.reconstructor(
            teacher_features, teacher_logits, student_logits,
            partition.high, partition.uncertain, coordinates=coordinates)
        output['loss_reconstruction'] = (
            output['loss_reconstruction'] * self.reconstruction_weight)
        output['loss_distillation'] = (
            output['loss_distillation'] * self.distillation_weight)
        output.update({
            'high_mask': partition.high,
            'uncertain_mask': partition.uncertain,
            'low_mask': partition.low,
            'gmm_posterior': partition.posterior,
            'joint_confidence': partition.joint_confidence,
            'teacher_student_consistency': partition.consistency,
        })
        return output
