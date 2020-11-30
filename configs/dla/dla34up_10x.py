_base_ = [
    '../_base_/models/dla34up.py',
    '../_base_/datasets/bdd100k.py',
]

img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train = dict(pipeline=train_pipeline),
    val = dict(pipeline=test_pipeline),
    test = dict(pipeline=test_pipeline))

# TODO lr changed!!!
optimizer = dict(
            type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
            policy='step',
            warmup='exp',
            warmup_iters=500,
            warmup_ratio=0.1 / 3,
            step=[80, 110])

total_epochs = 120
checkpoint_config = dict(interval=10)
log_config = dict(
            interval=10,
            hooks=[
                # dict(type='WandBLoggerHook', project_name='bdd-'),
                dict(type='TextLoggerHook'),
                # dict(type='TensorboardLoggerHook')
            ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = './work_dirs/debug/bdd'
resume_from = None
workflow = [('train', 10)]