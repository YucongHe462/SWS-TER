"""Framework-level integration smoke test for SWS-TER.

Run this file from the repository root with the legacy MMRotate environment::

    python tests/framework_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import platformdirs
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(tempfile.gettempdir()) / "sws-ter-framework-check"
RUNTIME.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

# Recent YAPF tries to create its grammar cache under the user profile while
# MMCV is imported. Keep its generated grammar cache outside the repository.
platformdirs.user_cache_dir = lambda *args, **kwargs: str(RUNTIME)

from mmcv import Config  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402


def _metadata(size=128):
    return [dict(
        filename="smoke.png",
        ori_filename="smoke.png",
        ori_shape=(size, size, 1),
        img_shape=(size, size, 1),
        pad_shape=(size, size, 1),
        scale_factor=1.0,
        flip=False,
        tag="unsup_weak_unlabeled")]


def _check_exact_sal_mapping(student):
    head = student.bbox_head
    with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as directory:
        label_path = Path(directory) / "smoke.npy"
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[:, 8:] = 1
        np.save(str(label_path), labels)
        head.sal_partitions = {
            "smoke": [
                dict(image_id="smoke", superpixel_id=0,
                     superpixel_map=str(label_path),
                     priors=dict(P_tar=1.0, P_bg=0.0, P_hard=0.0)),
                dict(image_id="smoke", superpixel_id=1,
                     superpixel_map=str(label_path),
                     priors=dict(P_tar=0.0, P_bg=1.0, P_hard=0.0)),
            ]
        }
        points = [torch.tensor([[2.0, 2.0], [12.0, 2.0]])]
        weights = head.build_sal_cls_weights(
            _metadata(16), points, num_imgs=1, device=torch.device("cpu"))
        # The head flattens RBox and HBox target sets, hence the second copy.
        expected = torch.tensor([0.05, 1.95, 0.05, 1.95])
        torch.testing.assert_close(weights, expected)
        support = student._region_support(
            torch.zeros(1, 1, 16, 16), _metadata(16))
        torch.testing.assert_close(support[0, 0, :, :8],
                                   torch.ones(16, 8))
        torch.testing.assert_close(support[0, 0, :, 8:],
                                   torch.zeros(16, 8))


def _check_eq33_gate(head):
    logits = torch.tensor([[-2.0], [2.0]])
    background = torch.tensor([1, 1])
    baseline = head.loss_cls(
        logits, background, reduction_override="none")
    weighted = head.loss_cls(
        logits, background, weight=torch.full((2,), 10.0),
        reduction_override="none")
    torch.testing.assert_close(weighted[0], baseline[0])
    torch.testing.assert_close(weighted[1], baseline[1] * 10.0)


def main():
    config = Config.fromfile(str(
        ROOT / "configs" / "sws_ter" /
        "sws_ter_Pd_Pv_Sa_hbox_20_20.py"))
    import_modules_from_strings(**config.custom_imports)
    # A smoke test must not download ImageNet weights.
    config.model.model.backbone.init_cfg = None
    model = build_detector(config.model)
    model.train()
    assert model.student.training
    assert not model.teacher.training
    student = model.student.eval()
    head = student.bbox_head

    assert model.inference_on == "student"
    assert head.loss_cent.loss_weight == 0.05
    assert head.loss_cls.thresh == 0.5
    assert head.loss_cls.hard_negative_weight == 0.4
    assert head.loss_cls.weight_hard_negatives_only
    assert len(model.semi_loss_unsup.recovery.reconstructor.reconstructor.layers) == 4

    with torch.no_grad():
        output = student.forward_train(
            torch.randn(1, 1, 128, 128), _metadata(), get_data=True)
    assert len(output[0]) == 5 and len(output[-1]) == 5
    assert all(torch.isfinite(item).all() for group in output[:4]
               for item in group)

    unsupervised = model.semi_loss_unsup(output, output)
    assert all(torch.isfinite(value) for key, value in unsupervised.items()
               if key.startswith("loss_"))
    _check_exact_sal_mapping(student)
    _check_eq33_gate(head)

    student_parameters = sum(parameter.numel()
                             for parameter in student.parameters())
    swcs_parameters = sum(parameter.numel()
                          for parameter in student.evidence_completion.parameters())
    mar_parameters = sum(parameter.numel()
                         for parameter in model.semi_loss_unsup.recovery.parameters())
    frozen_edge_parameters = sum(parameter.numel()
                                 for parameter in student.boundary_extractor.parameters())
    baseline_parameters = (student_parameters - swcs_parameters
                           - frozen_edge_parameters)
    table9 = [baseline_parameters,
              baseline_parameters + swcs_parameters,
              baseline_parameters + mar_parameters,
              baseline_parameters + swcs_parameters + mar_parameters]
    assert [round(value / 1e6, 1) for value in table9] == [
        32.1, 33.9, 34.3, 36.1]
    print("SWS-TER framework smoke test passed")
    print("Table 9 parameters (M):",
          [round(value / 1e6, 1) for value in table9])
    print("feature shapes:", [tuple(item.shape) for item in output[-1]])


if __name__ == "__main__":
    main()
