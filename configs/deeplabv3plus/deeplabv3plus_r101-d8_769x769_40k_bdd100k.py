_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/bdd100k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    )
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
