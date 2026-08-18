# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import torch

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from torchvision import transforms

from semi_mmrotate.models.sar_boundary import SARBoundaryExtractor

@ROTATED_DETECTORS.register_module()
class SWSterBaseDetector(RotatedSingleStageDetector):
    """Single-stage detector with mixed RBox, HBox and point supervision."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rotate_range = (0.25, 0.75),
                 scale_range = (0.5, 0.9),
                 ss_prob = [0.6, 0.15, 0.25],
                 train_cfg = None,
                 test_cfg = None,
                 pretrained=None,
                 init_cfg = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob

        self.boundary_extractor = SARBoundaryExtractor(
            with_uncertainty=True, roa_windows=[3, 5, 7])
        for parameter in self.boundary_extractor.parameters():
            parameter.requires_grad = False
        self.boundary_extractor.eval()

    def forward_train(self, 
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      get_data=False,
                      return_augmented_targets=False,
                      gt_bboxes_ignore=None):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            img_metas (list[dict]): The batch image metadata.
            gt_bboxes (list[Tensor]): Ground truth bounding boxes for each image.
            gt_labels (list[Tensor]): Class labels for each ground truth box.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes that are
                ignored during training. Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        H, W = img.shape[2:4]
        batch_gt_instances = []
        
        self.bbox_head.iter_count = self.iter_count
        
        # Convert gt_bboxes and gt_labels into structured instance format
        for i in range(len(gt_bboxes)):
            instance = {'bboxes': gt_bboxes[i], 'labels': gt_labels[i]}
            batch_gt_instances.append(instance)

        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances['bboxes'])
            bids = gt_instances['labels'].new_zeros(blen, 4)
            bids[:, 0] = i
            bids[:, 3] = torch.arange(0, blen, 1) + offset
            gt_instances['bids'] = bids
            offset += blen

        sel_p = torch.rand(1)
        if sel_p < self.ss_prob[0]:
            # Generate rotated images and gts
            rot = math.pi * (
                torch.rand(1).item() *
                (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            for meta in img_metas:
                meta['ss'] = ('rot', rot)
            img_aug = transforms.functional.rotate(img, -rot / math.pi * 180)
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :2] = (gt_instances['bboxes'][..., :2] - ctr).matmul(tf.T) + ctr
                gt_instances['bboxes'][:, 4] = gt_instances['bboxes'][:, 4] + rot
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            # Generate flipped images and gts
            for meta in img_metas:
                meta['ss'] = ('flp', 0)
            img_aug = transforms.functional.vflip(img)
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, 1] = img.shape[-2] - gt_instances['bboxes'][:, 1]
                gt_instances['bboxes'][:, 4] = -gt_instances['bboxes'][:, 4]
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        else:
            # Generate scaled images and gts
            sca = (torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            for meta in img_metas:
                meta['ss'] = ('sca', sca)
            img_aug = transforms.functional.resized_crop(img, 0, 0, int(H / sca), int(W / sca), [H, W])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :4] *= sca
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
                
        img_all = torch.cat((img, img_aug))
        self.bbox_head.images = img_all
        # SAR-native boundary evidence for the geometric loss.
        if self.iter_count >= self.bbox_head.edge_loss_start_iter:
            with torch.no_grad():
                # Denormalize back to original pixel intensity [0, 255]
                # SSDD images are grayscale stored as 3-ch, so we denormalize
                # then take single channel for SAR-native processing.
                if img_all.shape[1] == 1:
                    mean = img_all.new_tensor([114.5])[..., None, None]
                    std = img_all.new_tensor([57.9])[..., None, None]
                else:
                    mean = img_all.new_tensor(
                        [123.675, 116.28, 103.53])[..., None, None]
                    std = img_all.new_tensor(
                        [58.395, 57.12, 57.375])[..., None, None]
                img_denorm = img_all * std + mean  # Back to [0, 255]
                # Convert to single-channel SAR intensity
                sar_intensity = img_denorm.mean(dim=1, keepdim=True)  # (B,1,H,W)
                batch_edges = self.boundary_extractor(sar_intensity)
                self.bbox_head.edges = batch_edges[3].clamp(0)  # Fused boundary field
                # Pass uncertainty map to head for clutter-adaptive weighting
                if len(batch_edges) > 4:
                    self.bbox_head.edge_uncertainty = batch_edges[4]
                else:
                    self.bbox_head.edge_uncertainty = None

        batch_inputs_all = torch.cat((img, img_aug))
        batch_data_samples_all = []
        for gt_instances, img_metas in zip(batch_gt_instances + batch_gt_aug, img_metas + img_metas):
            data_sample = {'metainfo': img_metas, 'gt_instances': gt_instances}
            batch_data_samples_all.append(data_sample)
        feat = self.extract_feat(batch_inputs_all)
        cls_score, bbox_pred, angle_pred, centerness = self.bbox_head.forward(feat, get_data)
        
        batch_gt_instances = [copy.deepcopy(data_sample['gt_instances']) for data_sample in batch_data_samples_all]
        batch_img_metas = [data_sample['metainfo'] for data_sample in batch_data_samples_all]
          
        if return_augmented_targets:
            return (cls_score, bbox_pred, angle_pred, centerness, batch_gt_instances)

        if get_data:
            return (cls_score, bbox_pred, angle_pred, centerness,
                    getattr(self.bbox_head, 'spp_outputs', None))
        
        results_list = self.bbox_head.get_bboxes((cls_score[0],), 
                                                 (bbox_pred[0],), 
                                                 (angle_pred[0],),
                                                 (centerness[0],), 
                                                 batch_img_metas, 
                                                 batch_gt_instances=batch_gt_instances)
        converted_results_list = []
        for det_bboxes, det_labels in results_list:
            bboxes = det_bboxes[:, :5]
            scores = det_bboxes[:, 5]
            result_dict = {
                'bboxes': bboxes,
                'scores': scores,  
                'labels': det_labels 
            }
            converted_results_list.append(result_dict)
        
        # Update point annotations with predicted rbox
        for data_sample, results in zip(batch_gt_instances, converted_results_list):
            mask = data_sample['bids'][:, 1] == 0
            data_sample['bboxes'][mask] = results['bboxes'][mask]
            data_sample['labels'][mask] = results['labels'][mask]

        losses = self.bbox_head.loss(cls_score,
                                     bbox_pred,
                                     angle_pred,
                                     centerness,
                                     batch_gt_instances,
                                     batch_img_metas)

        return losses
