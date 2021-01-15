_base_ = [
    '../_base_/models/dla102up.py',
    '../_base_/datasets/bdd100k.py',
]

img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        flip=False,
        transforms=[
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train = dict(pipeline=train_pipeline),
    val = dict(pipeline=test_pipeline),
    test = dict(pipeline=test_pipeline))

# TODO lr changed!!!
optimizer = dict(
            type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
