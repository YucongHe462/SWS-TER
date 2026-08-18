"""Annotation-free contrastive prior construction (paper Eqs. 9--12)."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SCFEEncoder(nn.Module):
    """Three-block self-supervised contrastive feature encoder.

    Channel widths follow the manuscript exactly: 32, 64 and 128.  The
    representation mapping and two-layer projection head implement Eq. (10).
    """

    def __init__(self,
                 in_channels: int = 3,
                 representation_dim: int = 128,
                 projection_dim: int = 64) -> None:
        super().__init__()
        blocks = []
        current = in_channels
        for channels in (32, 64, 128):
            blocks.extend([
                nn.Conv2d(current, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            current = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.representation = nn.Linear(128, representation_dim)
        self.projector = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(representation_dim, projection_dim),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        feature = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.representation(feature), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        representation = self.forward_features(x)
        return F.normalize(self.projector(representation), dim=1)


class MomentumContrastiveEncoder(nn.Module):
    """Momentum SCFE with an InfoNCE dictionary.

    The key encoder is updated by Eq. (11); the queue-based objective is
    Eq. (12).  Calling :meth:`encode_regions` returns the normalized
    pre-projection representation used to form region prototypes.
    """

    def __init__(self,
                 in_channels: int = 3,
                 representation_dim: int = 128,
                 projection_dim: int = 64,
                 queue_size: int = 8192,
                 momentum: float = 0.999,
                 temperature: float = 0.07) -> None:
        super().__init__()
        if queue_size <= 0:
            raise ValueError('queue_size must be positive')
        self.encoder_q = SCFEEncoder(in_channels, representation_dim,
                                     projection_dim)
        self.encoder_k = deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad_(False)
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        queue = F.normalize(torch.randn(queue_size, projection_dim), dim=1)
        self.register_buffer('queue', queue)
        self.register_buffer('queue_pointer', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self) -> None:
        for query, key in zip(self.encoder_q.parameters(),
                              self.encoder_k.parameters()):
            key.mul_(self.momentum).add_(query,
                                         alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys: Tensor) -> None:
        keys = keys.detach()
        if keys.shape[0] >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size:])
            self.queue_pointer.zero_()
            return
        pointer = int(self.queue_pointer.item())
        end = pointer + keys.shape[0]
        if end <= self.queue_size:
            self.queue[pointer:end].copy_(keys)
        else:
            first = self.queue_size - pointer
            self.queue[pointer:].copy_(keys[:first])
            self.queue[:end - self.queue_size].copy_(keys[first:])
        self.queue_pointer[0] = end % self.queue_size

    def forward(self, query_view: Tensor, key_view: Tensor) -> Tensor:
        query = self.encoder_q(query_view)
        with torch.no_grad():
            self.momentum_update()
            key = self.encoder_k(key_view)
        positive = torch.einsum('nd,nd->n', query, key).unsqueeze(1)
        negative = torch.einsum('nd,kd->nk', query,
                                self.queue.detach().clone())
        logits = torch.cat((positive, negative), dim=1) / self.temperature
        labels = torch.zeros(query.shape[0], dtype=torch.long,
                             device=query.device)
        loss = F.cross_entropy(logits, labels)
        self._enqueue(key)
        return loss

    @torch.no_grad()
    def encode_regions(self, regions: Tensor, momentum_encoder: bool = False) -> Tensor:
        encoder = self.encoder_k if momentum_encoder else self.encoder_q
        return encoder.forward_features(regions)

