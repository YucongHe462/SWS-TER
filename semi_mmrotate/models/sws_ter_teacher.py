import torch
from .teacher_student_detector import TeacherStudentDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector

@ROTATED_DETECTORS.register_module()
class SWSterTeacher(TeacherStudentDetector):
    def __init__(self, model: dict, semi_loss_unsup, train_cfg=None, test_cfg=None):
        super().__init__(
            dict(teacher=build_detector(model), student=build_detector(model)), 
            semi_loss_unsup,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze('teacher')
            self.iter_count = train_cfg.get('iter_count', 0)
            self.burn_in_steps = train_cfg.get('burn_in_steps', 5000)
            self.sup_weight = train_cfg.get('sup_weight', 1.0)
            self.unsup_weight = train_cfg.get('unsup_weight', 1.0)

    def forward_train(self, imgs, img_metas, **kwargs):
        super().forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')

        # iter_count
        self.teacher.iter_count = self.iter_count
        self.student.iter_count = self.iter_count
        
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data:
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)

        losses = dict()
        if 'sup' in format_data:
            sup_losses = self.student.forward_train(**format_data['sup'])
            for key, val in sup_losses.items():
                if key[:4] == 'loss':
                    if isinstance(val, list):
                        losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                    else:
                        losses[f"{key}_sup"] = self.sup_weight * val
                else:
                    losses[key] = val
        if (self.iter_count >= self.burn_in_steps
                and 'unsup_weak_unlabeled' in format_data
                and 'unsup_strong_unlabeled' in format_data):
            with torch.no_grad():
                teacher_logits_unlabeled = self.teacher.forward_train(
                    get_data=True, **format_data['unsup_weak_unlabeled'])
            student_logits_unlabeled = self.student.forward_train(get_data=True,  **format_data['unsup_strong_unlabeled'])
            
            unsup_losses_unlabeled = self.semi_loss_unsup(
                teacher_logits_unlabeled,
                student_logits_unlabeled,
                img_metas=format_data['unsup_weak_unlabeled'],
                iter_count=self.iter_count)

            for key, val in unsup_losses_unlabeled.items():
                if 'loss' in key:
                    losses[f"{key}_unsup_unlabeled"] = self.unsup_weight * val
                else:
                    losses[key] = val

        return losses
