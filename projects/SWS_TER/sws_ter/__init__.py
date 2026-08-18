"""SWS-TER model components.

The pure PyTorch components can be imported without MMDetection/MMRotate.
Framework adapters are registered only when the legacy OpenMMLab stack is
available.
"""

from .models import (EvidenceGuidedContextScatteringFusion,
                     MomentumContrastiveEncoder,
                     PriorGuidedScatteringKeypointGraph,
                     PrototypeGuidedReconstructor,
                     ScaleAdaptiveContextCompensation,
                     SparseWeakEvidenceCompletion,
                     TorchGMM2,
                     UncertaintyGuidedRecovery)

try:  # Optional legacy MMRotate integration.
    from .models.detectors import SWSterStudent  # noqa: F401
    from .models.ugsrt_loss import UncertaintyGuidedRecoveryLoss  # noqa: F401
except ImportError:
    pass

__all__ = [
    'EvidenceGuidedContextScatteringFusion',
    'MomentumContrastiveEncoder',
    'PriorGuidedScatteringKeypointGraph',
    'PrototypeGuidedReconstructor',
    'ScaleAdaptiveContextCompensation',
    'SparseWeakEvidenceCompletion',
    'TorchGMM2',
    'UncertaintyGuidedRecovery',
]

