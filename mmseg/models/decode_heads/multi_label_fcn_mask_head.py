import mmcv
import numpy as np
# import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..builder import HEADS
from ..model_utils import ConvModule
from mmseg.core import mask_target

# replace mmdet.core with mmcv.runner
from mmcv.runner import auto_fp16, force_fp32
# from .decode_head import BaseDecodeHead


@HEADS.register_module
class MultiLabelFCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=4,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 fg_weight=1,
                 name='',
                 ignore_index=255,
                 align_corners=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(MultiLabelFCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes if type(num_classes) == list else [num_classes]
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.name = name
        self.fg_weight = fg_weight
        # add new features
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = [1 if self.class_agnostic else c for c in self.num_classes]
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.ModuleList([nn.Conv2d(logits_in_channel, out_channel, 1) for out_channel in out_channels])
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        modules = [*self.conv_logits]
        if self.upsample_method == 'deconv':
            modules.append(self.upsample)
        for m in modules:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = [l(x) for l in self.conv_logits]
        return mask_pred

    # def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
    #     pos_proposals = [res.pos_bboxes for res in sampling_results]
    #     pos_assigned_gt_inds = [
    #         res.pos_assigned_gt_inds for res in sampling_results
    #     ]
    #     mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
    #                                gt_masks, rcnn_train_cfg)
    #     return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def losses(self, mask_pred, mask_targets):
        loss = dict()
        for i, (p, t, c) in enumerate(zip(mask_pred, mask_targets, self.num_classes)):
            if len(t) == 0:
                continue
            # apply foreground weights
            # TODO: move this elsewhere
            weights = None
            if self.fg_weight > 1:
                weights = torch.FloatTensor([1] + [self.fg_weight] * (c - 1)).to(p.device)
            loss_mask = self.loss_mask(p, t, class_weight=weights, ignore_index=self.ignore_index)
            loss['loss_{}'.format(self.name)] = loss_mask
        return loss
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    
