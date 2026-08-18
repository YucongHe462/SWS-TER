# Copyright (c) OpenMMLab. All rights reserved.
"""Weak geometric losses for mixed RBox, HBox and point supervision."""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weighted_loss

from mmrotate.models.builder import ROTATED_LOSSES, build_roi_extractor
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred,
                   target,
                   fun='log1p',
                   tau=1.0,
                   alpha=1.0,
                   normalize=True):
    """Gaussian Wasserstein distance loss for covariance matrices only."""
    sigma_p = pred
    sigma_t = target

    whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    trace_term = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2,
                                                 dim2=-1).sum(dim=-1)
    det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance - 2 * (
        (trace_term + 2 * det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(
            1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate Bhattacharyya coefficient between 2-D Gaussians."""
    xy_p, sigma_p = pred
    xy_t, sigma_t = target

    shape = xy_p.shape
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    sigma_m = (sigma_p + sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    coef = torch.exp(-0.125 *
                     dxy.permute(0, 2, 1).bmm(torch.linalg.solve(
                         sigma_m, dxy)))
    det_pair = (sigma_p.det() * sigma_t.det()).clamp(1e-7).sqrt()
    coef = coef * (det_pair / sigma_m.det()).clamp(1e-7).sqrt()[..., None,
                                                                 None]
    return coef.reshape(shape[:-1])


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
    """Penalize excessive overlap between Gaussian instance supports."""
    del target
    mu, sigma = pred
    num_inst = mu.shape[0]
    mu0 = mu[None].expand(num_inst, num_inst, 2)
    sigma0 = sigma[None].expand(num_inst, num_inst, 2, 2)
    mu1 = mu[:, None].expand(num_inst, num_inst, 2)
    sigma1 = sigma[:, None].expand(num_inst, num_inst, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(num_inst, dtype=torch.bool, device=loss.device)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    return loss.sum(-1)


@ROTATED_LOSSES.register_module()
class GaussianOverlapLoss(nn.Module):
    """Penalize overlap between Gaussian instance supports."""

    def __init__(self, reduction='mean', loss_weight=1.0, lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        eigvals = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(eigvals, torch.zeros_like(eigvals),
                              reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()

        loss_overlap = gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * (loss_lamb + loss_overlap)


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    prob = torch.exp(-0.5 *
                     dxy.permute(0, 2, 1).bmm(torch.linalg.solve(
                         sigma, dxy)))
    if normalize:
        prob = prob / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return prob


# ===========================================================================
# SR-SVW: Speckle-Resilient Statistical Voronoi Watershed
# ===========================================================================
# Physics-informed utility functions for SAR-adapted watershed segmentation.
#
# Key insight: Under SAR's multiplicative noise model (I = sigma0 * n),
# the Coefficient of Variation (CoV = std/mean) is a sufficient statistic
# for edge detection. For homogeneous regions, CoV = 1/sqrt(L) (constant),
# while at true scattering boundaries CoV significantly increases.
# This provides a CFAR (Constant False Alarm Rate) edge metric that is
# inherently robust to speckle noise.
# ===========================================================================

def compute_sar_heterogeneity_field(image, kernel_size=7):
    """
    Compute the statistical heterogeneity field for SAR imagery.

    Based on the Coefficient of Variation (CoV) which is a sufficient
    statistic for edge detection under the multiplicative noise model.

    Theory:
        Under H0 (homogeneous clutter): CoV = 1/sqrt(L), constant
        Under H1 (scattering boundary): CoV >> 1/sqrt(L)
        ENL (Equivalent Number of Looks) = 1/CoV^2

    Note: Input images may be ImageNet-normalized (mostly negative values).
    This function first restores positive intensity via denormalization,
    then computes CoV on the positive-valued SAR intensity.

    Args:
        image: (C, H, W) tensor, SAR image (possibly normalized).
        kernel_size (int): Local statistics window size. Default: 7.
            Larger windows give more stable estimates but less localization.

    Returns:
        cov_map: (H, W) Coefficient of Variation map (edge indicator).
        enl_map: (H, W) Equivalent Number of Looks map (noise level indicator).
    """
    # Convert to single-channel intensity
    img = image.mean(0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

    # Handle normalized images: restore to positive intensity domain
    # If image appears to be normalized (has significant negative values),
    # apply inverse normalization to recover original SAR intensity
    if img.min() < -0.5:
        # ImageNet denormalization for single-channel: approx mean=114.5, std=57.9
        img = img * 57.9 + 114.5
    # Ensure positivity for CoV computation (SAR intensity is non-negative)
    img = img.clamp(min=1e-3)

    pad = kernel_size // 2

    # Local statistics via efficient average pooling
    local_mean = F.avg_pool2d(img, kernel_size, stride=1, padding=pad)
    local_sq_mean = F.avg_pool2d(img ** 2, kernel_size, stride=1, padding=pad)
    local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-7)

    # CoV = sqrt(Var) / |Mean| — CFAR edge metric
    cov_map = local_var.sqrt() / (local_mean.abs() + 1e-7)

    # ENL estimation: L ≈ Mean^2 / Var = 1 / CoV^2
    enl_map = 1.0 / (cov_map ** 2 + 1e-7)

    return cov_map.squeeze(0).squeeze(0), enl_map.squeeze(0).squeeze(0)


def gaussian_voronoi_watershed_loss(mu,
                                    sigma,
                                    label,
                                    image,
                                    pos_thres,
                                    neg_thres,
                                    down_sample=2,
                                    topk=0.95,
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    sar_mode=False,
                                    enl_adaptive=True,
                                    cov_kernel_size=7):
    """
    Fit Gaussian instance extent to watershed regions.

    SR-SVW Enhancement (when sar_mode=True):
        Replaces optical gradient-based watershed with a statistical
        heterogeneity field (CoV) driven watershed, and introduces
        ENL-adaptive confidence modulation for Voronoi partition thresholds.

    Args:
        mu: (N, 2) Instance center coordinates.
        sigma: (N, 2, 2) Instance covariance matrices.
        label: (N,) Instance class labels.
        image: (C, H, W) Input image tensor.
        pos_thres: Positive threshold per class.
        neg_thres: Negative threshold per class.
        down_sample (int): Voronoi computation downsampling factor.
        topk (float): Top-k ratio for loss computation.
        default_sigma (float): Default Gaussian spread for Voronoi.
        voronoi (str): Voronoi partition type.
        alpha (float): Unused legacy parameter.
        sar_mode (bool): Enable SAR-specific SR-SVW mode.
        enl_adaptive (bool): Enable ENL-adaptive threshold modulation.
        cov_kernel_size (int): Kernel size for CoV computation.
    """
    del alpha
    num_inst = len(sigma)
    if num_inst == 0:
        return sigma.sum(), None

    down = down_sample
    height, width = image.shape[-2:]
    small_h, small_w = height // down, width // down
    x = torch.linspace(0, small_h, small_h, device=mu.device)
    y = torch.linspace(0, small_w, small_w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(num_inst, small_h, small_w)

    centers = (mu.detach() / down).round()
    if voronoi == 'standard':
        base_sigma = sigma.new_tensor((default_sigma, 0, 0,
                                       default_sigma)).reshape(2, 2)
        base_sigma = base_sigma / down**2
        for idx, center in enumerate(centers):
            vor[idx] = gaussian_2d(xy.view(-1, 2), center[None],
                                   base_sigma[None]).view(small_h, small_w)
    elif voronoi == 'gaussian-orientation':
        eigvals, eigvecs = torch.linalg.eigh(sigma)
        eigvals = eigvals.detach().clone()
        eigvals = eigvals / (eigvals[:, 0:1] * eigvals[:, 1:2]).sqrt(
        ) * default_sigma
        oriented_sigma = eigvecs.matmul(torch.diag_embed(eigvals)).matmul(
            eigvecs.permute(0, 2, 1)).detach()
        oriented_sigma = oriented_sigma / down**2
        for idx, (center, cov) in enumerate(zip(centers, oriented_sigma)):
            vor[idx] = gaussian_2d(xy.view(-1, 2), center[None],
                                   cov[None]).view(small_h, small_w)
    elif voronoi == 'gaussian-full':
        full_sigma = sigma.detach() / down**2
        for idx, (center, cov) in enumerate(zip(centers, full_sigma)):
            vor[idx] = gaussian_2d(xy.view(-1, 2), center[None],
                                   cov[None]).view(small_h, small_w)
    else:
        raise ValueError(f'Unsupported voronoi type: {voronoi}')

    val, vor = torch.max(vor, 0)
    if down > 1:
        vor = vor[:, None, :, None].expand(-1, down, -1,
                                           down).reshape(height, width)
        val = F.interpolate(
            val[None, None], (height, width), mode='bilinear',
            align_corners=True)[0, 0]

    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0

    vor = vor + 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)

    # === SR-SVW: ENL-Adaptive Confidence Modulation ===
    # High ENL (low noise) -> tighten thresholds (more confident)
    # Low ENL (high noise) -> relax thresholds (more conservative)
    # Formula: tau = tau_0 * L_hat / (L_hat + 1)
    if sar_mode and enl_adaptive:
        _, enl_map = compute_sar_heterogeneity_field(image, cov_kernel_size)
        mean_enl = enl_map.mean().clamp(min=1.0)
        enl_factor = (mean_enl / (mean_enl + 1.0)).clamp(0.5, 1.5)
        pos_thres = pos_thres * enl_factor
        neg_thres = neg_thres * enl_factor

    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = num_inst + 1
    vor[ridges] = num_inst + 1

    # === SR-SVW: Statistical Heterogeneity Field Watershed ===
    if sar_mode:
        # Compute CoV-based heterogeneity field as watershed input
        # CoV is a CFAR-theoretic edge indicator: constant under H0 (homogeneous
        # clutter), significantly elevated at true scattering boundaries.
        # This replaces optical Sobel gradients which fail under speckle.
        cov_map, _ = compute_sar_heterogeneity_field(image, cov_kernel_size)

        # Normalize CoV field to uint8 for OpenCV watershed
        # High CoV -> high "gradient" -> watershed cuts here
        cov_np = cov_map.detach().cpu().numpy()
        cov_min, cov_max = cov_np.min(), cov_np.max()
        cov_norm = (cov_np - cov_min) / (cov_max - cov_min + 1e-7)
        cov_uint8 = (cov_norm * 255).astype(np.uint8)
        # OpenCV watershed requires 3-channel input
        cov_3ch = np.stack([cov_uint8] * 3, axis=-1)
        # Light smoothing on CoV field to suppress residual noise
        cov_3ch = cv2.medianBlur(cov_3ch, 3)

        markers = vor.detach().cpu().numpy().astype(np.int32)
        markers = vor.new_tensor(cv2.watershed(cov_3ch, markers))
    else:
        # Original optical mode (backward compatible)
        img_uint8 = (image - image.min()) / (image.max() - image.min() + 1e-7)
        img_uint8 = img_uint8 * 255
        img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(
            np.uint8)
        img_uint8 = cv2.medianBlur(img_uint8, 3)
        markers = vor.detach().cpu().numpy().astype(np.int32)
        markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

    eigvals, eigvecs = torch.linalg.eigh(sigma)
    eig_targets = []
    for idx in range(num_inst):
        region_xy = (markers == idx + 1).nonzero()[:, (1, 0)].float()
        if len(region_xy) == 0:
            eig_targets.append(eigvals[idx].detach())
            continue
        region_xy = region_xy - mu[idx]
        region_xy = eigvecs[idx].T.matmul(region_xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(region_xy[:, 0]))
        max_y = torch.max(torch.abs(region_xy[:, 1]))
        eig_targets.append(torch.stack((max_x, max_y))**2)

    eig_targets = torch.diag_embed(torch.stack(eig_targets))
    eigvals = torch.diag_embed(eigvals)
    loss = gwd_sigma_loss(eigvals, eig_targets.detach(), reduction='none')
    loss = torch.topk(
        loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss


@ROTATED_LOSSES.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Voronoi watershed loss.

    SR-SVW Enhancement (when sar_mode=True):
        - Replaces optical gradient watershed with CoV-driven watershed
        - Introduces ENL-adaptive threshold modulation
        - Provides CFAR-theoretic robustness to multiplicative speckle noise
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 sar_mode=False,
                 enl_adaptive=True,
                 cov_kernel_size=7):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.sar_mode = sar_mode
        self.enl_adaptive = enl_adaptive
        self.cov_kernel_size = cov_kernel_size

    def forward(self, pred, label, image, pos_thres, neg_thres,
                voronoi='orientation'):
        loss = gaussian_voronoi_watershed_loss(
            *pred,
            label,
            image,
            pos_thres,
            neg_thres,
            self.down_sample,
            topk=self.topk,
            voronoi=voronoi,
            alpha=self.alpha,
            sar_mode=self.sar_mode,
            enl_adaptive=self.enl_adaptive,
            cov_kernel_size=self.cov_kernel_size)
        return self.loss_weight * loss


@ROTATED_LOSSES.register_module()
class GaussianVoronoiLoss(VoronoiWatershedLoss):
    """Configured name for the Voronoi watershed loss."""


def rbbox2roi(bbox_list):
    """Convert a list of rotated bboxes to RoI format."""
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    return torch.cat(rois_list, 0)


@ROTATED_LOSSES.register_module()
class EdgeLoss(nn.Module):
    """Edge-guided size refinement loss.

    SAR Enhancement:
        When uncertainty map is provided, edge responses in high-uncertainty
        (heavy speckle / low SCR) regions are suppressed before computing
        the size refinement target. This prevents noisy edge responses from
        corrupting bbox size estimation.

    Args:
        resolution (int): RoI feature resolution. Default: 24.
        max_scale (float): RoI expansion factor. Default: 1.6.
        sigma (float): Gaussian distribution sigma for expected edge position. Default: 6.
        reduction (str): Loss reduction mode. Default: 'mean'.
        loss_weight (float): Loss weight. Default: 1.0.
        uncertainty_suppression (bool): Whether to use uncertainty map for
            suppressing unreliable edge responses. Default: True.
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 uncertainty_suppression=True):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.uncertainty_suppression = uncertainty_suppression

        self.roi_extractor = build_roi_extractor(
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
                out_channels=1,
                featmap_strides=[1],
                finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx)**2) /
                                      (2 * self.sigma**2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge, uncertainty=None):
        """
        Args:
            pred: List of rotated bboxes per image.
            edge: (B, 1, H, W) Edge/boundary field map.
            uncertainty: (B, 1, H, W) Optional boundary uncertainty map.
                Values in [0, 1], higher = less reliable.
        """
        rois = rbbox2roi(pred)
        if rois.size(0) == 0:
            return edge.new_tensor(0)

        # === SAR: Uncertainty-suppressed edge field ===
        # In high-uncertainty regions (heavy speckle), edge responses are
        # attenuated to prevent noisy gradients from corrupting size estimation
        if self.uncertainty_suppression and uncertainty is not None:
            edge = edge * (1.0 - uncertainty)

        grid = self.resolution
        center = self.center_idx
        rois[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], rois)
        if len(feat) == 0:
            return edge.new_tensor(0)

        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :grid + 1], (-1, )) + featx[:, grid:]
        featy2 = torch.flip(featy[:, :grid + 1], (-1, )) + featy[:, grid:]
        ex = ((featx2 * self.edge_distribution).softmax(1) *
              self.edge_idx).sum(1) / center
        ey = ((featy2 * self.edge_distribution).softmax(1) *
              self.edge_idx).sum(1) / center
        edge_scale = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)

        return self.loss_weight * F.smooth_l1_loss(
            rbbox_concat[:, 2:4],
            (rbbox_concat[:, 2:4] * edge_scale).detach(),
            beta=8)
