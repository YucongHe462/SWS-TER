from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feat_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.representation = nn.Linear(128, feat_dim)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        h = self.representation(x)
        return F.normalize(h, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 128,
        projection_dim: int = 64,
        queue_size: int = 8192,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder_q = PatchEncoder(in_channels, feat_dim, projection_dim)
        self.encoder_k = PatchEncoder(in_channels, feat_dim, projection_dim)
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, projection_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :])
            self.queue_ptr[0] = 0
            return
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end] = keys
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end - self.queue_size] = keys[first:]
        self.queue_ptr[0] = end % self.queue_size

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> torch.Tensor:
        q = self.encoder_q(im_q)
        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(im_k)

        positive = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
        negative = torch.einsum("nc,kc->nk", q, self.queue.clone().detach())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self.dequeue_and_enqueue(k)
        return loss


def build_model(args) -> ContrastiveModel:
    return ContrastiveModel(
        in_channels=args.in_channels,
        feat_dim=args.feat_dim,
        projection_dim=args.projection_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
    )
