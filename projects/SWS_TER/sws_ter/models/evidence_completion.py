"""Sparse-weak evidence completion student feature path."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from torch import Tensor, nn

from .egcsf import EvidenceGuidedContextScatteringFusion
from .pskg import PriorGuidedScatteringKeypointGraph
from .sacc import ScaleAdaptiveContextCompensation


class SparseWeakEvidenceCompletion(nn.Module):
    """Apply SACC, PSKG and EGCSF to every FPN level."""

    def __init__(self,
                 channels: int = 256,
                 num_levels: int = 5,
                 sacc: Optional[dict] = None,
                 pskg: Optional[dict] = None,
                 egcsf: Optional[dict] = None,
                 share_sacc: bool = False,
                 share_fusion: bool = False) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        sacc = dict(sacc or {})
        pskg = dict(pskg or {})
        egcsf = dict(egcsf or {})
        if share_sacc:
            module = ScaleAdaptiveContextCompensation(channels, **sacc)
            self.sacc = nn.ModuleList([module] * self.num_levels)
        else:
            self.sacc = nn.ModuleList([
                ScaleAdaptiveContextCompensation(channels, **sacc)
                for _ in range(self.num_levels)
            ])
        self.pskg = PriorGuidedScatteringKeypointGraph(channels, **pskg)
        if share_fusion:
            module = EvidenceGuidedContextScatteringFusion(channels, **egcsf)
            self.fusion = nn.ModuleList([module] * self.num_levels)
        else:
            self.fusion = nn.ModuleList([
                EvidenceGuidedContextScatteringFusion(channels, **egcsf)
                for _ in range(self.num_levels)
            ])

    def forward(self,
                features: Sequence[Tensor],
                sar: Tensor,
                support_prior: Optional[Tensor] = None
                ) -> Tuple[List[Tensor], Dict[str, object]]:
        if len(features) != self.num_levels:
            raise ValueError(f'expected {self.num_levels} FPN levels, got {len(features)}')
        contexts, scale_weights = [], []
        for module, feature in zip(self.sacc, features):
            context, weights = module(feature)
            contexts.append(context)
            scale_weights.append(weights)
        structural, evidence, graph_diagnostics = self.pskg(
            features, sar, support_prior=support_prior)
        enhanced, reliability = [], []
        for module, original, context, graph_feature, graph_evidence in zip(
                self.fusion, features, contexts, structural, evidence):
            output, gate = module(original, context, graph_feature,
                                  graph_evidence)
            enhanced.append(output)
            reliability.append(gate)
        auxiliary: Dict[str, object] = {
            'context_features': contexts,
            'scale_weights': scale_weights,
            'structural_features': structural,
            'structural_evidence': evidence,
            'structural_reliability': reliability,
            'graph_diagnostics': graph_diagnostics,
        }
        return enhanced, auxiliary

