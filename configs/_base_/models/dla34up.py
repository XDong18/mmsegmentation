# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='/shared/xudongliu/code/weights/dla34-ba72cf86.pth',
    backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block_num=2,
        return_levels=True),
    neck=dict(
        type='DLAUp',
        in_channels=[32, 64, 128, 256, 512],
        channels=[256, 256, 256, 256, 256],
        scales=(1, 2, 4, 8, 16),
        num_outs=5),
    decode_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=19,
            name='sem_seg',
            loss_mask=dict(
                type='CrossEntropyLoss'),
            upsample_method='bilinear')
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')