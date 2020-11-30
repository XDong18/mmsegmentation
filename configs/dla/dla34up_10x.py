_base_ = [
    'configs/_base_/models/dla34up.py',
    '../_base_/datasets/bdd100k.py',
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)


img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(
            type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

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
                dict(type='WandBLoggerHook', project_name='bdd-'),
                dict(type='TextLoggerHook')
                # dict(type='TensorboardLoggerHook')
            ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = './work_dirs/debug/bdd'
resume_from = None
workflow = [('train', 10)]