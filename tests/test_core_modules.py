import torch
import numpy as np

from projects.SWS_TER.sws_ter.models.acpc import MomentumContrastiveEncoder
from projects.SWS_TER.sws_ter.models.evidence_completion import SparseWeakEvidenceCompletion
from projects.SWS_TER.sws_ter.models.losses import (region_prior_weight,
                                                    sparse_prior_focal_loss)
from projects.SWS_TER.sws_ter.models.sacc import ScaleAdaptiveContextCompensation
from projects.SWS_TER.sws_ter.models.pskg import PriorGuidedScatteringKeypointGraph
from projects.SWS_TER.sws_ter.models.teacher import (PrototypeGuidedReconstructor,
                                                     TorchGMM2,
                                                     UncertaintyGuidedRecovery)
from tools.prepare_covariance import pauli_coherency


def test_pauli_coherency_is_hermitian():
    hh = np.ones((3, 4), dtype=np.complex64) * (1 + 2j)
    hv = np.ones((3, 4), dtype=np.complex64) * (0.5 - 0.2j)
    vv = np.ones((3, 4), dtype=np.complex64) * (2 - 1j)
    matrix = pauli_coherency(hh, hv, vv)
    assert matrix.shape == (3, 4, 3, 3)
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2).conj())
    vector = np.asarray([1 + 0.5j, -0.2j, 0.7], dtype=np.complex64)
    quadratic = np.einsum('i,...ij,j->...', vector.conj(), matrix, vector)
    assert np.all(quadratic.real >= -1e-5)
    np.testing.assert_allclose(quadratic.imag, 0, atol=1e-5)


def test_scfe_info_nce_and_queue_update():
    torch.manual_seed(1)
    model = MomentumContrastiveEncoder(
        in_channels=3, representation_dim=16, projection_dim=8,
        queue_size=16, momentum=0.9)
    query = torch.rand(4, 3, 32, 32)
    key = torch.rand(4, 3, 32, 32)
    old_pointer = int(model.queue_pointer)
    loss = model(query, key)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert int(model.queue_pointer) == (old_pointer + 4) % 16
    loss.backward()
    assert model.encoder_q.features[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder_k.parameters())


def test_sacc_competitive_weights():
    module = ScaleAdaptiveContextCompensation(channels=16)
    feature = torch.rand(2, 16, 16, 16, requires_grad=True)
    context, weights = module(feature)
    assert context.shape == feature.shape
    assert weights.shape == (2, 3, 16, 16)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    context.mean().backward()
    assert feature.grad is not None


def test_complete_student_feature_path():
    torch.manual_seed(2)
    module = SparseWeakEvidenceCompletion(
        channels=16,
        num_levels=2,
        pskg=dict(node_channels=8, graph_layers=1, topk=8, knn=3),
        egcsf=dict(residual_initial_value=0.1))
    features = [torch.rand(2, 16, 16, 16, requires_grad=True),
                torch.rand(2, 16, 8, 8, requires_grad=True)]
    sar = torch.rand(2, 1, 64, 64)
    support = torch.ones(2, 1, 64, 64)
    output, auxiliary = module(features, sar, support)
    assert [item.shape for item in output] == [item.shape for item in features]
    assert len(auxiliary['structural_evidence']) == 2
    assert all(torch.isfinite(item).all() for item in output)
    sum(item.mean() for item in output).backward()
    assert all(item.grad is not None for item in features)


def test_pskg_splat_preserves_edge_keypoints():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    coordinates = torch.tensor([[3, 1], [2, 3]])
    output = PriorGuidedScatteringKeypointGraph._bilinear_splat(
        values, coordinates, height=4, width=4)
    torch.testing.assert_close(output[:, 3, 1], values[0])
    torch.testing.assert_close(output[:, 2, 3], values[1])


def test_pskg_eq27_aggregates_cluster_evidence_by_sum():
    first = torch.ones(4, 5)
    second = torch.full((4, 5), 2.0)
    third = torch.full((4, 5), 4.0)
    evidence = PriorGuidedScatteringKeypointGraph._aggregate_cluster_evidence(
        [first, second, third])
    torch.testing.assert_close(evidence, torch.full((4, 5), 7.0))


def test_eq32_and_sparse_focal_only_modulate_hard_negatives():
    weight = region_prior_weight(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(weight, torch.tensor([0.05, 1.95, 1.85]))
    logits = torch.tensor([[2.0], [2.0], [-2.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0], [0.0]])
    loss = sparse_prior_focal_loss(logits, targets, weight,
                                   hard_negative_threshold=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_eq33_sal_weight_is_restricted_to_high_confidence_negatives():
    logits = torch.tensor([[-2.0], [2.0]])
    targets = torch.zeros_like(logits)
    baseline = sparse_prior_focal_loss(
        logits, targets, reduction='none', hard_negative_threshold=0.5,
        hard_negative_modulation=1.0)
    weighted = sparse_prior_focal_loss(
        logits, targets, prior_weight=torch.full((2,), 10.0),
        reduction='none', hard_negative_threshold=0.5,
        hard_negative_modulation=1.0)
    torch.testing.assert_close(weighted[0], baseline[0])
    torch.testing.assert_close(weighted[1], baseline[1] * 10.0)


def test_torch_gmm_returns_high_component_posterior():
    torch.manual_seed(3)
    low = torch.randn(64) * 0.02 + 0.1
    high = torch.randn(64) * 0.02 + 0.8
    scores = torch.cat((low, high))
    posterior = TorchGMM2().posterior_high(scores)
    assert posterior[64:].mean() > 0.95
    assert posterior[:64].mean() < 0.05


def test_latent_prototype_reconstruction():
    torch.manual_seed(4)
    module = PrototypeGuidedReconstructor(
        feature_dim=16, num_classes=2, embed_dim=16, num_layers=1,
        num_heads=4, max_context=8, max_uncertain=4)
    teacher_features = torch.rand(12, 16)
    teacher_logits = torch.full((12, 2), -3.0)
    teacher_logits[:4, 0] = 4.0
    teacher_logits[4:8, 1] = 4.0
    teacher_logits[8:, 0] = 0.2
    student_logits = torch.randn(12, 2, requires_grad=True)
    high = torch.zeros(12, dtype=torch.bool)
    high[:8] = True
    uncertain = ~high
    coordinates = torch.stack((torch.arange(12), torch.arange(12)), dim=1)
    output = module(teacher_features, teacher_logits, student_logits,
                    high, uncertain, coordinates)
    assert output['soft_labels'].shape == (4, 2)
    assert module.prototype_initialized.all()
    total = output['loss_reconstruction'] + output['loss_distillation']
    assert torch.isfinite(total)
    total.backward()
    assert student_logits.grad is not None


def test_level_specific_uncertainty_recovery_masks_are_disjoint():
    torch.manual_seed(5)
    module = UncertaintyGuidedRecovery(
        feature_dim=8, num_classes=1,
        reconstructor=dict(embed_dim=8, num_layers=1, num_heads=2,
                           max_context=16, max_uncertain=8))
    count = 24
    teacher_features = torch.rand(count, 8)
    teacher_logits = torch.linspace(-4, 4, count).reshape(-1, 1)
    teacher_center = torch.linspace(-2, 3, count).reshape(-1, 1)
    student_logits = teacher_logits.detach().clone().requires_grad_(True)
    output = module(teacher_features, teacher_logits, teacher_center,
                    student_logits, [slice(0, 12), slice(12, 24)])
    total = (output['high_mask'].int() + output['uncertain_mask'].int()
             + output['low_mask'].int())
    assert torch.equal(total, torch.ones_like(total))


def test_teacher_student_consistency_uses_class_and_joint_confidence():
    module = UncertaintyGuidedRecovery(
        feature_dim=8, num_classes=2,
        reconstructor=dict(embed_dim=8, num_layers=1, num_heads=2,
                           max_context=8, max_uncertain=4))
    teacher_logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    teacher_center = torch.tensor([[4.0], [4.0]])
    matching_logits = teacher_logits.clone()
    mismatching_logits = torch.tensor([[-5.0, 5.0], [-5.0, 5.0]])
    student_center = teacher_center.clone()
    matching = module.partition(
        teacher_logits, teacher_center, matching_logits, [slice(0, 2)],
        student_center)
    mismatching = module.partition(
        teacher_logits, teacher_center, mismatching_logits, [slice(0, 2)],
        student_center)
    assert matching.consistency.mean() > 0.99
    torch.testing.assert_close(mismatching.consistency,
                               torch.zeros_like(mismatching.consistency))
