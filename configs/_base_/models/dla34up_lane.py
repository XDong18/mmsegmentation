# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='Multi_head_EncoderDecoder',
    pretrained='/shared/xudongliu/models/dla34-24a49e58.pth',
    backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block_num=2,
        num_classes=19,
        return_levels=True,
        norm_cfg=norm_cfg
        ),
    neck=dict(
        type='DLAUp',
        channels=[32, 64, 128, 256, 512],
        scales=(1, 2, 4, 8, 16),
        norm_cfg=norm_cfg
        ),
    lane_dir_head=dict(
            type='DLAsHead',
            channels=32,
            in_channels=32,
            num_classes=3,
            dropout_ratio=0,
            loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=255),
            ),
        lane_sty_head=dict(
            type='DLAsHead',
            channels=32,
            in_channels=32,
            num_classes=3,
            dropout_ratio=0,
            loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=255),
            ),
        lane_typ_head=dict(
            type='DLAsHead',
            channels=32,
            in_channels=32,
            num_classes=9,
            dropout_ratio=0,
            loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=255),
            )
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
