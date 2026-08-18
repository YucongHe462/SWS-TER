"""Scale-adaptive context compensation (paper Eqs. 13--19)."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn


class _ContextBranch(nn.Sequential):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size, padding=padding,
                      dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class _ZPool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((x.amax(dim=1, keepdim=True),
                          x.mean(dim=1, keepdim=True)), dim=1)


class _AttentionGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.compress = _ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x: Tensor) -> Tensor:
        # Eq. (16) defines A_l^v as the restored attention response itself;
        # feature weighting happens later through the branch selector.
        return torch.sigmoid(self.norm(self.conv(self.compress(x))))


class _TriAxisInteraction(nn.Module):
    """Height-channel, width-channel and spatial interaction.

    Dimension permutation followed by Z-pooling implements the three views in
    Eq. (16).  All three outputs are restored to NCHW before Eq. (17).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.height_channel = _AttentionGate()
        self.width_channel = _AttentionGate()
        self.spatial = _AttentionGate()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, channels, height, width = x.shape
        # A_hc: restore as N,C,H,1 and broadcast along width.
        hc = x.permute(0, 3, 2, 1).contiguous()
        hc = self.height_channel(hc).permute(0, 3, 2, 1).contiguous()
        hc = hc.expand(-1, channels, height, width)
        # A_wc: restore as N,C,1,W and broadcast along height.
        wc = x.permute(0, 2, 1, 3).contiguous()
        wc = self.width_channel(wc).permute(0, 2, 1, 3).contiguous()
        wc = wc.expand(-1, channels, height, width)
        # A_sp is shared across channels in Eq. (16).
        spatial = self.spatial(x).expand(-1, channels, height, width)
        return self.fuse(torch.cat((hc, wc, spatial), dim=1))


class ScaleAdaptiveContextCompensation(nn.Module):
    """Three-range competitive context compensation module."""

    def __init__(self,
                 channels: int = 256,
                 kernels: Sequence[int] = (3, 5, 7),
                 dilations: Sequence[int] = (1, 2, 3),
                 reduction: int = 16) -> None:
        super().__init__()
        if len(kernels) != 3 or len(dilations) != 3:
            raise ValueError('SACC requires exactly three context branches')
        if reduction <= 0:
            raise ValueError('SACC reduction must be positive')
        self.branches = nn.ModuleList([
            _ContextBranch(channels, int(kernel), int(dilation))
            for kernel, dilation in zip(kernels, dilations)
        ])
        # Eq. (17) forms a compact unified descriptor rather than another
        # full-width FPN tensor. C/16 also reproduces the SWCS parameter count
        # reported in manuscript Table 9 (33.9M versus the 32.1M baseline).
        descriptor_channels = max(channels // int(reduction), 1)
        self.tri_axis = _TriAxisInteraction(
            3 * channels, descriptor_channels)
        self.selector = nn.Conv2d(descriptor_channels, 3, 1)

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        branches = [branch(feature) for branch in self.branches]
        descriptor = self.tri_axis(torch.cat(branches, dim=1))
        weights = torch.softmax(self.selector(descriptor), dim=1)
        context = sum(branch * weights[:, index:index + 1]
                      for index, branch in enumerate(branches))
        return context, weights
