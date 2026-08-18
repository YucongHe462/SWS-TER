"""Legacy MMRotate adapter for the SWS-TER student branch."""

from __future__ import annotations

import math
import os.path as osp
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from mmrotate.models.builder import ROTATED_DETECTORS
from semi_mmrotate.models.detectors.sws_ter_base import SWSterBaseDetector

from .evidence_completion import SparseWeakEvidenceCompletion


@ROTATED_DETECTORS.register_module()
class SWSterStudent(SWSterBaseDetector):
    """Sparse weak detector with SACC, PSKG and EGCSF.

    The inherited head retains the RBox/HBox/Point geometric losses.  This
    adapter only replaces the FPN feature path and exposes latent features to
    UGSRT; it does not alter the detector's box parameterization.
    """

    def __init__(self,
                 *args,
                 evidence_completion: Optional[dict] = None,
                 input_mean=(114.5,),
                 input_std=(57.9,),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = dict(evidence_completion or {})
        config.setdefault('channels', 256)
        config.setdefault('num_levels', 5)
        self.evidence_completion = SparseWeakEvidenceCompletion(**config)
        self.register_buffer(
            '_input_mean', torch.tensor(input_mean).reshape(1, -1, 1, 1))
        self.register_buffer(
            '_input_std', torch.tensor(input_std).reshape(1, -1, 1, 1))
        self._pending_support: Optional[Tensor] = None
        self._pending_img_metas = None
        self._last_evidence_features = None
        self._last_evidence_aux = None

    def _region_support(self, images: Tensor, img_metas) -> Optional[Tensor]:
        partitions = getattr(self.bbox_head, 'sal_partitions', None)
        if not partitions or img_metas is None:
            return None
        _, _, height, width = images.shape
        support = images.new_zeros((len(img_metas), 1, height, width))
        yy, xx = torch.meshgrid(torch.arange(height, device=images.device),
                                torch.arange(width, device=images.device),
                                indexing='ij')
        for image_index, meta in enumerate(img_metas):
            filename = meta.get('ori_filename') or meta.get('filename') or ''
            items = partitions.get(osp.splitext(osp.basename(filename))[0], [])
            map_path = items[0].get('superpixel_map') if items else None
            if map_path:
                prior_json = getattr(self.bbox_head, 'sal_prior_json', None)
                if not osp.isabs(map_path) and prior_json:
                    map_path = osp.join(osp.dirname(prior_json), map_path)
                if osp.exists(map_path):
                    cache = getattr(self.bbox_head, '_sal_label_cache', None)
                    if cache is not None and map_path in cache:
                        raw_labels = cache[map_path]
                    else:
                        raw_labels = torch.from_numpy(
                            np.load(map_path).astype(np.int64))
                        if cache is not None:
                            cache[map_path] = raw_labels
                    encoded = (raw_labels.to(images.device) + 1).float()
                    image_height, image_width = meta['img_shape'][:2]
                    labels = F.interpolate(
                        encoded[None, None],
                        (image_height, image_width), mode='nearest')[0, 0]
                    labels = labels.round().long() - 1
                    if meta.get('flip', False):
                        direction = meta.get('flip_direction', 'horizontal')
                        if direction in ('horizontal', 'diagonal'):
                            labels = torch.flip(labels, dims=(1,))
                        if direction in ('vertical', 'diagonal'):
                            labels = torch.flip(labels, dims=(0,))
                    for item in items:
                        prior = item.get('priors', item)
                        target_probability = float(prior.get(
                            'P_tar', prior.get(
                                'p_target', prior.get('sim_target', 0.0))))
                        region = labels == int(item.get('superpixel_id', -1))
                        support[image_index, 0, :image_height,
                                :image_width][region] = target_probability
                    continue

            # Backward compatibility for old ACPC JSON files without the
            # persisted integer superpixel map.
            scale = meta.get('scale_factor', 1.0)
            if hasattr(scale, '__len__'):
                sx, sy = float(scale[0]), float(scale[1])
            else:
                sx = sy = float(scale)
            for item in items:
                prior = item.get('priors', item)
                target_probability = float(prior.get(
                    'P_tar', prior.get('p_target', prior.get('sim_target', 0.0))))
                if target_probability <= 0:
                    continue
                x, y = float(item['cx']) * sx, float(item['cy']) * sy
                if meta.get('flip', False):
                    direction = meta.get('flip_direction', 'horizontal')
                    if direction in ('horizontal', 'diagonal'):
                        x = width - x
                    if direction in ('vertical', 'diagonal'):
                        y = height - y
                sigma = max(float(item.get('crop_size', 24.0)) * max(sx, sy) / 4, 2.0)
                gaussian = torch.exp(-((xx - x).square() + (yy - y).square())
                                     / (2 * sigma * sigma))
                support[image_index, 0] = torch.maximum(
                    support[image_index, 0], target_probability * gaussian)
        return support

    def extract_feat(self, img: Tensor):
        # The released SAR tiles are single-channel.  Replication is only a
        # compatibility adapter for the ImageNet-pretrained 3-channel ResNet
        # stem; all SAR evidence is computed from the original one channel.
        if img.shape[1] == 1:
            backbone_input = img.repeat(1, 3, 1, 1)
        elif img.shape[1] == 3:
            backbone_input = img
        else:
            raise ValueError(f'SWS-TER expects 1 or 3 input channels, got {img.shape[1]}')
        features = self.backbone(backbone_input)
        if self.with_neck:
            features = self.neck(features)
        support = self._pending_support
        if support is not None and support.shape[0] != img.shape[0]:
            if img.shape[0] != 2 * support.shape[0]:
                raise ValueError('ACPC support batch cannot be aligned to augmented images')
            # The base detector appends a self-supervised geometric view. Apply
            # identical transform to the ACPC prior instead of merely
            # repeating it, otherwise PSKG would be guided to stale pixels.
            transform = self._pending_img_metas[0].get('ss', None)
            if transform is None:
                raise ValueError('missing self-supervised transform metadata')
            kind, value = transform
            if kind == 'rot':
                support_aug = transforms.functional.rotate(
                    support, -float(value) / math.pi * 180)
            elif kind == 'flp':
                support_aug = transforms.functional.vflip(support)
            elif kind == 'sca':
                height, width = support.shape[-2:]
                support_aug = transforms.functional.resized_crop(
                    support, 0, 0, int(height / float(value)),
                    int(width / float(value)), [height, width])
            else:
                raise ValueError(f'unknown self-supervised transform: {kind}')
            support = torch.cat((support, support_aug), dim=0)
        mean = self._input_mean.to(img)
        std = self._input_std.to(img)
        if mean.shape[1] not in (1, img.shape[1]):
            raise ValueError('input normalization channels do not match the image')
        sar = img * std + mean
        enhanced, auxiliary = self.evidence_completion(
            features, sar.mean(dim=1, keepdim=True), support_prior=support)
        self._last_evidence_features = enhanced
        self._last_evidence_aux = auxiliary
        return enhanced

    def forward_train(self, img, img_metas, *args, get_data=False, **kwargs):
        self._pending_support = self._region_support(img, img_metas)
        self._pending_img_metas = img_metas
        try:
            if get_data:
                # Teacher weak and student strong views already share their
                # geometric pipeline. The supervised path would independently sample a
                # second random rotation/flip/scale for each branch, making
                # dense distillation locations inconsistent.  The UGSRT path
                # therefore uses exactly one aligned view per image.
                self.bbox_head.iter_count = getattr(self, 'iter_count', 0)
                self.bbox_head.images = img
                features = self.extract_feat(img)
                predictions = self.bbox_head.forward(features, True)
                output = tuple(predictions) + (
                    getattr(self.bbox_head, 'spp_outputs', None), features)
            else:
                output = super().forward_train(
                    img, img_metas, *args, get_data=False, **kwargs)
        finally:
            self._pending_support = None
            self._pending_img_metas = None
        return output
