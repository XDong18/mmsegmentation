# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
            policy='step',
            warmup='exp',
            warmup_iters=500,
            warmup_ratio=0.1 / 3,
            step=[8, 11])
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=20000)
total_epochs = 12
checkpoint_config = dict(interval=1)
evaluation = dict(interval=2000, metric='mIoU')