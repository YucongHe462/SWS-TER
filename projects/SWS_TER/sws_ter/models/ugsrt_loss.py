"""MMRotate loss adapter for uncertainty-guided supervision recovery."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder
from mmrotate.models import ROTATED_LOSSES

from .losses import quality_focal_distillation
from .teacher import UncertaintyGuidedRecovery


@ROTATED_LOSSES.register_module()
class UncertaintyGuidedRecoveryLoss(nn.Module):
    """Dense high-confidence distillation plus latent-feature recovery.

    Reconstruction is performed on FPN latent features (256-D by default).
    Candidate GMMs are fitted independently for every FPN level.
    """

    def __init__(self,
                 feature_dim: int = 256,
                 cls_channels: int = 1,
                 angle_coder=None,
                 strides=(8, 16, 32, 64, 128),
                 recovery_cfg=None) -> None:
        super().__init__()
        self.cls_channels = int(cls_channels)
        self.angle_coder = build_bbox_coder(angle_coder or dict(
            type='PSCCoder', angle_version='le90', dual_freq=False,
            num_step=3, thr_mod=0))
        self.prior_generator = MlvlPointGenerator(list(strides))
        config = dict(recovery_cfg or {})
        self.recovery = UncertaintyGuidedRecovery(
            feature_dim=feature_dim,
            num_classes=self.cls_channels,
            **config)

    def _preprocess_predictions(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits[:4]
        if not (len(cls_scores) == len(bbox_preds) == len(angle_preds)
                == len(centernesses)):
            raise ValueError('prediction branches must have equal FPN levels')
        batch_size = cls_scores[0].shape[0]
        decoded_angles = []
        for angle_pred in angle_preds:
            level_angles = []
            for image_index in range(batch_size):
                flattened = angle_pred[image_index].permute(
                    1, 2, 0).reshape(-1, self.angle_coder.encode_size)
                level_angles.append(self.angle_coder.decode(
                    flattened, keepdim=True).detach())
            decoded_angles.append(torch.stack(level_angles, dim=0))

        predictions = []
        for image_index in range(batch_size):
            classification = torch.cat([
                level[image_index].permute(1, 2, 0).reshape(
                    -1, self.cls_channels) for level in cls_scores
            ], dim=0)
            boxes = torch.cat([
                torch.cat((box[image_index].permute(1, 2, 0).reshape(-1, 4),
                           angle[image_index]), dim=-1)
                for box, angle in zip(bbox_preds, decoded_angles)
            ], dim=0)
            centerness = torch.cat([
                level[image_index].permute(1, 2, 0).reshape(-1, 1)
                for level in centernesses
            ], dim=0)
            predictions.append((classification, boxes, centerness))
        return predictions

    @staticmethod
    def _flatten_features(features: Sequence[Tensor], image_index: int) -> Tensor:
        return torch.cat([
            level[image_index].permute(1, 2, 0).reshape(-1, level.shape[1])
            for level in features
        ], dim=0)

    @staticmethod
    def _level_slices(featmap_sizes) -> List[slice]:
        output, start = [], 0
        for height, width in featmap_sizes:
            end = start + int(height) * int(width)
            output.append(slice(start, end))
            start = end
        return output

    def _loss_one(self,
                  teacher,
                  student,
                  teacher_feature: Tensor,
                  level_slices: Sequence[slice],
                  coordinates: Tensor):
        teacher_cls, teacher_box, teacher_center = teacher
        student_cls, student_box, student_center = student
        recovery = self.recovery(
            teacher_feature,
            teacher_cls,
            teacher_center,
            student_cls,
            level_slices,
            coordinates=coordinates,
            student_centerness=student_center)
        high = recovery['high_mask']
        confidence = recovery['joint_confidence'].detach()
        weights = high.to(confidence.dtype) * confidence
        if high.any():
            loss_cls = quality_focal_distillation(
                student_cls, teacher_cls.detach().sigmoid(), weights)
            box_error = F.smooth_l1_loss(
                student_box[high], teacher_box.detach()[high], reduction='none')
            loss_box = (box_error.mean(dim=1) * weights[high]).sum() / (
                weights[high].sum().clamp_min(1e-6))
            center_error = F.binary_cross_entropy_with_logits(
                student_center[high], teacher_center.detach()[high].sigmoid(),
                reduction='none').reshape(-1)
            loss_center = (center_error * weights[high]).sum() / (
                weights[high].sum().clamp_min(1e-6))
        else:
            zero = student_cls.sum() * 0.0
            loss_cls = loss_box = loss_center = zero
        return (loss_cls, loss_box, loss_center,
                recovery['loss_reconstruction'],
                recovery['loss_distillation'],
                high.float().mean(), recovery['uncertain_mask'].float().mean())

    def forward(self, teacher_logits, student_logits, img_metas=None, **kwargs):
        del img_metas, kwargs
        if len(teacher_logits) < 6 or teacher_logits[5] is None:
            raise ValueError(
                'UGSRT requires latent FPN features from SWSterStudent')
        teacher_by_image = self._preprocess_predictions(teacher_logits)
        student_by_image = self._preprocess_predictions(student_logits)
        features = teacher_logits[5]
        featmap_sizes = [level.shape[-2:] for level in teacher_logits[0]]
        level_slices = self._level_slices(featmap_sizes)
        points = torch.cat(self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=teacher_logits[0][0].dtype,
            device=teacher_logits[0][0].device), dim=0)
        outputs = []
        for image_index, (teacher, student) in enumerate(
                zip(teacher_by_image, student_by_image)):
            outputs.append(self._loss_one(
                teacher, student,
                self._flatten_features(features, image_index),
                level_slices, points))
        count = max(len(outputs), 1)
        return {
            'loss_cls': sum(item[0] for item in outputs) / count,
            'loss_bbox': sum(item[1] for item in outputs) / count,
            'loss_centerness': sum(item[2] for item in outputs) / count,
            'loss_reconstruction': sum(item[3] for item in outputs) / count,
            'loss_distillation': sum(item[4] for item in outputs) / count,
            'ugsrt_high_ratio': sum(item[5] for item in outputs) / count,
            'ugsrt_uncertain_ratio': sum(item[6] for item in outputs) / count,
        }
