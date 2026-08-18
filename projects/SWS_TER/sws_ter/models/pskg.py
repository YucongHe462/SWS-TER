"""Prior-guided scattering keypoint graph (paper Eqs. 20--28)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel(size: int, sigma: float) -> Tensor:
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


class _GraphSAGELayer(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.update = nn.Linear(2 * dimensions, dimensions)
        self.norm = nn.LayerNorm(dimensions)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neighbourhood = adjacency @ nodes / degree
        updated = F.relu(self.update(torch.cat((nodes, neighbourhood), dim=1)))
        return self.norm(updated)


class PriorGuidedScatteringKeypointGraph(nn.Module):
    """Extract structure-tensor keypoints and reason with GraphSAGE.

    Keypoint selection and graph topology are intentionally non-differentiable;
    sampled FPN features and every learned graph/fusion layer remain fully
    differentiable.  ``support_prior`` is a soft ship-support map from ACPC.
    """

    def __init__(self,
                 in_channels: int = 256,
                 node_channels: int = 128,
                 graph_layers: int = 2,
                 topk: int = 64,
                 knn: int = 6,
                 response_balance: float = 0.04,
                 relative_threshold: float = 0.01,
                 spatial_sigma: float = 4.0,
                 response_sigma: float = 0.25,
                 cluster_radius: float = 6.0,
                 evidence_sigma: float = 2.5,
                 gaussian_size: int = 5,
                 gaussian_sigma: float = 1.0) -> None:
        super().__init__()
        self.topk = int(topk)
        self.knn = int(knn)
        self.response_balance = float(response_balance)
        self.relative_threshold = float(relative_threshold)
        self.spatial_sigma = float(spatial_sigma)
        self.response_sigma = float(response_sigma)
        self.cluster_radius = float(cluster_radius)
        self.evidence_sigma = float(evidence_sigma)
        self.node_projection = nn.Sequential(
            nn.Linear(in_channels + 3, node_channels),
            nn.ReLU(inplace=True),
        )
        self.graph_layers = nn.ModuleList(
            [_GraphSAGELayer(node_channels) for _ in range(graph_layers)])
        self.structure_classifier = nn.Sequential(
            nn.Linear(node_channels, node_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(node_channels // 2, 1),
        )
        self.node_to_feature = nn.Linear(node_channels, in_channels)
        self.register_buffer('sobel_x', torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
                             .reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
                             .reshape(1, 1, 3, 3))
        self.register_buffer('gaussian', _gaussian_kernel(
            gaussian_size, gaussian_sigma).reshape(1, 1, gaussian_size,
                                                    gaussian_size))

    def structure_response(self, sar: Tensor) -> Tensor:
        """Harris-like scattering response from Eqs. (20)--(21)."""
        if sar.ndim != 4:
            raise ValueError('sar must have shape [N,C,H,W]')
        intensity = sar.mean(dim=1, keepdim=True)
        flat = intensity.flatten(2)
        lo = flat.amin(dim=2, keepdim=True).unsqueeze(-1)
        hi = flat.amax(dim=2, keepdim=True).unsqueeze(-1)
        intensity = (intensity - lo) / (hi - lo).clamp_min(1e-6)
        grad_x = F.conv2d(intensity, self.sobel_x, padding=1)
        grad_y = F.conv2d(intensity, self.sobel_y, padding=1)
        padding = self.gaussian.shape[-1] // 2
        jxx = F.conv2d(grad_x.square(), self.gaussian, padding=padding)
        jxy = F.conv2d(grad_x * grad_y, self.gaussian, padding=padding)
        jyy = F.conv2d(grad_y.square(), self.gaussian, padding=padding)
        determinant = jxx * jyy - jxy.square()
        trace = jxx + jyy
        response = (determinant - self.response_balance * trace.square())
        response = response.clamp_min(0)
        maximum = response.flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1)
        return response / maximum.clamp_min(1e-6)

    def _select_keypoints(self, response: Tensor) -> List[Tuple[Tensor, Tensor]]:
        pooled = F.max_pool2d(response, 3, stride=1, padding=1)
        local_maximum = response >= pooled
        selected: List[Tuple[Tensor, Tensor]] = []
        for batch_index in range(response.shape[0]):
            values = response[batch_index, 0]
            keep = local_maximum[batch_index, 0]
            maximum = values.max()
            keep &= values >= maximum * self.relative_threshold
            coords = keep.nonzero(as_tuple=False)  # y, x
            scores = values[keep]
            if coords.numel() == 0:
                flat_index = values.argmax()
                width = values.shape[1]
                coords = torch.stack((flat_index // width,
                                      flat_index % width)).reshape(1, 2)
                scores = values.flatten()[flat_index].reshape(1)
            count = min(self.topk, scores.numel())
            order = scores.topk(count).indices
            selected.append((coords[order], scores[order]))
        return selected

    @staticmethod
    def _sample_features(feature: Tensor, coords_yx: Tensor) -> Tensor:
        height, width = feature.shape[-2:]
        y = coords_yx[:, 0].float()
        x = coords_yx[:, 1].float()
        grid_x = 2.0 * x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * y / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(feature.unsqueeze(0), grid, mode='bilinear',
                                align_corners=True)
        return sampled[0, :, :, 0].transpose(0, 1)

    def _adjacency(self, coords_yx: Tensor, scores: Tensor) -> Tensor:
        count = coords_yx.shape[0]
        if count == 1:
            return scores.new_ones((1, 1))
        positions = coords_yx.float()
        distance = torch.cdist(positions, positions)
        response_delta = (scores[:, None] - scores[None, :]).abs()
        affinity = torch.exp(
            -distance.square() / (2 * self.spatial_sigma**2)
            -response_delta.square() / (2 * self.response_sigma**2))
        affinity.fill_diagonal_(0)
        neighbours = min(self.knn, count - 1)
        indices = distance.masked_fill(
            torch.eye(count, dtype=torch.bool, device=distance.device),
            float('inf')).topk(neighbours, largest=False).indices
        mask = torch.zeros_like(affinity, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        mask = mask | mask.transpose(0, 1)
        return affinity * mask

    def _components(self, coords_yx: Tensor) -> List[Tensor]:
        """Spatial connected components used as candidate ship graphs."""
        count = coords_yx.shape[0]
        close = torch.cdist(coords_yx.float(), coords_yx.float()) <= self.cluster_radius
        visited = torch.zeros(count, dtype=torch.bool, device=coords_yx.device)
        components: List[Tensor] = []
        for start in range(count):
            if bool(visited[start]):
                continue
            frontier = [start]
            members = []
            visited[start] = True
            while frontier:
                current = frontier.pop()
                members.append(current)
                neighbours = (close[current] & ~visited).nonzero(
                    as_tuple=False).flatten().tolist()
                for neighbour in neighbours:
                    visited[neighbour] = True
                    frontier.append(neighbour)
            components.append(torch.tensor(members, device=coords_yx.device,
                                           dtype=torch.long))
        return components

    @staticmethod
    def _aggregate_cluster_evidence(cluster_maps: Sequence[Tensor]) -> Tensor:
        """Sum the cluster evidence fields as defined in Eq. (27)."""
        if not cluster_maps:
            raise ValueError('at least one cluster evidence map is required')
        return torch.stack(tuple(cluster_maps), dim=0).sum(dim=0)

    @staticmethod
    def _bilinear_splat(values: Tensor, coords_yx: Tensor,
                        height: int, width: int) -> Tensor:
        channels = values.shape[1]
        output = values.new_zeros((channels, height * width))
        normalizer = values.new_zeros((1, height * width))
        y = coords_yx[:, 0].float().clamp(0, height - 1)
        x = coords_yx[:, 1].float().clamp(0, width - 1)
        y0, x0 = y.floor().long(), x.floor().long()
        y1, x1 = (y0 + 1).clamp(max=height - 1), (x0 + 1).clamp(max=width - 1)
        lower_y = 1.0 - (y - y0.float())
        upper_y = y - y0.float()
        lower_x = 1.0 - (x - x0.float())
        upper_x = x - x0.float()
        # At the last row/column the upper and lower integer coordinates
        # coincide.  Assign the complete mass to that single coordinate.
        lower_y = torch.where(y1 == y0, torch.ones_like(lower_y), lower_y)
        upper_y = torch.where(y1 == y0, torch.zeros_like(upper_y), upper_y)
        lower_x = torch.where(x1 == x0, torch.ones_like(lower_x), lower_x)
        upper_x = torch.where(x1 == x0, torch.zeros_like(upper_x), upper_x)
        for yy, xx, weight in (
                (y0, x0, lower_y * lower_x),
                (y0, x1, lower_y * upper_x),
                (y1, x0, upper_y * lower_x),
                (y1, x1, upper_y * upper_x)):
            index = yy * width + xx
            output.scatter_add_(1, index[None].expand(channels, -1),
                                values.transpose(0, 1) * weight[None])
            normalizer.scatter_add_(1, index[None], weight[None])
        return (output / normalizer.clamp_min(1e-6)).reshape(channels,
                                                              height, width)

    def _one_level(self, feature: Tensor, response: Tensor) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
        batch, _, height, width = feature.shape
        response = F.interpolate(response, (height, width), mode='bilinear',
                                 align_corners=False)
        keypoints = self._select_keypoints(response)
        struct_batches = []
        evidence_batches = []
        diagnostics: List[Dict[str, Tensor]] = []
        yy, xx = torch.meshgrid(torch.arange(height, device=feature.device),
                                torch.arange(width, device=feature.device),
                                indexing='ij')
        for batch_index, (coords, scores) in enumerate(keypoints):
            semantic = self._sample_features(feature[batch_index], coords)
            position = torch.stack((coords[:, 1].float() / max(width - 1, 1),
                                    coords[:, 0].float() / max(height - 1, 1)),
                                   dim=1)
            nodes = self.node_projection(torch.cat((scores[:, None], position,
                                                     semantic), dim=1))
            adjacency = self._adjacency(coords, scores)
            for layer in self.graph_layers:
                nodes = layer(nodes, adjacency)
            mapped = self.node_to_feature(nodes)
            struct_batches.append(self._bilinear_splat(
                mapped, coords, height, width))

            cluster_confidences = []
            cluster_maps = []
            for component in self._components(coords):
                component_nodes = nodes[component]
                confidence = torch.sigmoid(
                    self.structure_classifier(component_nodes.mean(0))).squeeze()
                weights = scores[component].clamp_min(1e-6)
                center = (coords[component].float() * weights[:, None]).sum(0) / weights.sum()
                gaussian = torch.exp(-((yy - center[0]).square()
                                       + (xx - center[1]).square())
                                     / (2 * self.evidence_sigma**2))
                cluster_maps.append(confidence * gaussian)
                cluster_confidences.append(confidence)
            evidence_batches.append(
                self._aggregate_cluster_evidence(cluster_maps))
            diagnostics.append({
                'coordinates_yx': coords.detach(),
                'responses': scores.detach(),
                'cluster_confidences': torch.stack(cluster_confidences).detach(),
            })
        return (torch.stack(struct_batches, dim=0),
                torch.stack(evidence_batches, dim=0)[:, None], diagnostics)

    def forward(self,
                features: Sequence[Tensor],
                sar: Tensor,
                support_prior: Optional[Tensor] = None
                ) -> Tuple[List[Tensor], List[Tensor], List[List[Dict[str, Tensor]]]]:
        response = self.structure_response(sar)
        if support_prior is not None:
            if support_prior.ndim == 3:
                support_prior = support_prior[:, None]
            support = F.interpolate(support_prior.float(), response.shape[-2:],
                                    mode='bilinear', align_corners=False)
            response = response * support.clamp(0, 1)
        structural, evidence, diagnostics = [], [], []
        for feature in features:
            level_struct, level_evidence, level_diagnostics = self._one_level(
                feature, response)
            structural.append(level_struct)
            evidence.append(level_evidence)
            diagnostics.append(level_diagnostics)
        return structural, evidence, diagnostics
