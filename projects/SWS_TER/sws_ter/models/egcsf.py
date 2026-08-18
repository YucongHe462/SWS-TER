"""Evidence-guided context-scattering fusion (paper Eqs. 29--31)."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class EvidenceGuidedContextScatteringFusion(nn.Module):
    def __init__(self, channels: int = 256,
                 residual_initial_value: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Conv2d(3, 1, 3, padding=1)
        self.structural_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_initial_value)))

    def forward(self, original: Tensor, context: Tensor, structural: Tensor,
                evidence: Tensor) -> Tuple[Tensor, Tensor]:
        gate_input = torch.cat((evidence,
                                context.mean(dim=1, keepdim=True),
                                structural.amax(dim=1, keepdim=True)), dim=1)
        reliability = torch.sigmoid(self.gate(gate_input))
        fused = context + reliability * self.structural_refinement(structural)
        enhanced = original + self.residual_scale * fused
        return enhanced, reliability

