# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='/shared/xudongliu/models/dla102_27a30eac.pth',
    backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 3, 4, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_num=1,
        return_levels=True),
    neck=dict(
        type='DLAUp',
        channels=[32, 64, 128, 256, 512],
        scales=(1, 2, 4, 8, 16)
        ),
    decode_head=dict(
            type='DLAsHead',
            channels=1024,
            in_channels=1024,
            num_classes=19,
            dropout_ratio=0,
            loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
            )
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
