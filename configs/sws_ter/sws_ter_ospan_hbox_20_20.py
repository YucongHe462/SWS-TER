angle_version = 'le90'

import os

custom_imports = dict(
    imports=['semi_mmrotate', 'projects.SWS_TER.sws_ter'],
    allow_failed_imports=False)

import torchvision.transforms as transforms
from copy import deepcopy

data_root = os.path.abspath(os.environ.get(
    'SWS_TER_DATA_ROOT', './data/ospan_sws_ter')).replace('\\', '/') + '/'
acpc_prior_file = os.path.abspath(os.environ.get(
    'SWS_TER_ACPC_PRIOR', './work_dirs/acpc/acpc_priors.json')).replace(
        '\\', '/')

detector = dict(
    type='SWSterStudent',
    ss_prob=[0.68, 0.07, 0.25],
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='SWSterHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=False,
        edge_loss_start_iter=0,
        voronoi_type='standard',
        voronoi_thres=dict(default=[0.994, 0.005]),
        square_cls=[],
        edge_loss_cls=[0],
        post_process={},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cls=dict(
            type='SparseFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            # Eq. (33): only high-confidence negatives p(a)>thr receive the
            # 0.4 suppression and the SAL region-prior multiplier.
            thresh=0.5,
            loss_weight=1.0,
            hard_negative_weight=0.4,
            weight_hard_negatives_only=True),
        # Manuscript Eq. (43): lambda_ctr = 0.05.
        loss_cent=dict(type='mmdet.L1Loss', loss_weight=0.05),
        loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(
            type='GaussianVoronoiLoss',
            loss_weight=5.0,
            sar_mode=True,
            enl_adaptive=True,
            cov_kernel_size=7),
        loss_bbox_edg=dict(
            type='EdgeLoss',
            loss_weight=0.3,
            uncertainty_suppression=True),
        loss_ss=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
        gwd_weight=1.0,
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        sal_prior_json=acpc_prior_file,
        sal_assign_radius=24.0),
    evidence_completion=dict(
        channels=256,
        num_levels=5,
        sacc=dict(kernels=(3, 5, 7), dilations=(1, 2, 3), reduction=16),
        pskg=dict(
            node_channels=128,
            graph_layers=2,
            topk=64,
            knn=6,
            response_balance=0.04,
            relative_threshold=0.01),
        egcsf=dict(residual_initial_value=0.1)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

model = dict(
    type='SWSterTeacher',
    model=detector,
    semi_loss_unsup=dict(
        type='UncertaintyGuidedRecoveryLoss',
        cls_channels=1,
        feature_dim=256,
        recovery_cfg=dict(
            high_posterior=0.70,
            low_posterior=0.20,
            high_confidence=0.20,
            uncertain_confidence=0.02,
            high_consistency=0.50,
            uncertain_consistency=0.20,
            reconstruction_weight=0.5,
            distillation_weight=0.5,
            reconstructor=dict(
                embed_dim=256,
                # Four layers reproduce the +2.2M MAR cost in Table 9.
                num_layers=4,
                num_heads=8,
                mlp_ratio=2.0,
                prototype_momentum=0.99,
                max_context=256,
                max_uncertain=128))),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=12800,
        sup_weight=1.0,
        # Stage III uses L_total = L_sup + L_unsup (Eq. 46) after burn-in.
        unsup_weight=1.0),
    # The manuscript explicitly retains the student branch for inference.
    test_cfg=dict(inference_on='student'))

img_norm_cfg = dict(mean=[114.5], std=[57.9], to_rgb=False)
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'tag'))
]
strong_pipeline_unlabeled = [
    dict(type='SARToPILImage'),
    dict(
        type='SARRandomApply',
        operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8),
    dict(type='SARRandomGrayscale', p=0.2),
    dict(
        type='SARRandomApply',
        operations=[dict(type='SARGaussianBlur', rad_range=[0.1, 2.0])]),
    dict(type='SARToNumpy'),
    dict(type='ExtraAttrs', tag='unsup_strong_unlabeled'),
]
weak_pipeline_unlabeled = [
    dict(type='RResize', img_scale=(800, 800), ratio_range=(0.8, 1.2)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='ExtraAttrs', tag='unsup_weak_unlabeled'),
]
unsup_pipeline_unlabeled = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='TeacherStudentMultiBranch',
        unsup_strong=deepcopy(strong_pipeline_unlabeled),
        unsup_weak=deepcopy(weak_pipeline_unlabeled),
        common_pipeline=common_pipeline,
        is_seq=True),
]
sup_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ConvertSparseAnnotations',
        point_proportion=0.0,
        # OSPAN source annotations are horizontal boxes.
        hbox_proportion=1.0,
        modify_labels=True,
        version=angle_version),
    dict(type='RResize', img_scale=(800, 800), ratio_range=(0.8, 1.2)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='ExtraAttrs', tag='sup_weak'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'tag'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'SARDataset'
classes = ('ship', )
img_suffix = '.jpg'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='SparseDataset',
        sup=dict(
            type=dataset_type,
            ann_file=data_root + 'semi_ratio_20/sparse_ratio_20/label_annotation',
            img_prefix=data_root + 'semi_ratio_20/sparse_ratio_20/label_image',
            img_suffix=img_suffix,
            classes=classes,
            pipeline=sup_pipeline),
        unsup_unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'semi_ratio_20/sparse_ratio_20/unlabel_annotation',
            img_prefix=data_root + 'semi_ratio_20/sparse_ratio_20/unlabel_image',
            img_suffix=img_suffix,
            classes=classes,
            pipeline=unsup_pipeline_unlabeled,
            filter_empty_gt=False)),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'test_image',
        ann_file=data_root + 'test_annotation',
        img_suffix=img_suffix,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'test_image',
        ann_file=data_root + 'test_annotation',
        img_suffix=img_suffix,
        classes=classes,
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(type='MultiSourceSampler', sample_ratio=[1, 1], seed=42)))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MeanTeacher', momentum=0.9996, interval=1, start_steps=12800),
]

evaluation = dict(
    type='SubModulesDistEvalHook', interval=3200, metric='mAP', save_best='mAP')

optimizer = dict(
    type='Adam', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3)
runner = dict(type='IterBasedRunner', max_iters=48000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=3)

seed = 42
deterministic = True

log_config = dict(
    _delete_=True, interval=50, hooks=[dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
