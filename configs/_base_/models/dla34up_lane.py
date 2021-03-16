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
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            name='lane_dir',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear'),
        lane_sty_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            name='lane_sty',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear'),
        lane_typ_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            name='lane_typ',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear')
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
