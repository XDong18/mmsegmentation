import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import math
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

@HEADS.register_module()
class DLAsHead(BaseDecodeHead):
    def __init__(self, channels, fg_weight=1, **kwargs):
        self.channels = channels
        self.first_level = 1
        self.fg_weight = fg_weight
        super(DLAsHead, self).__init__(channels=channels, **kwargs)
        # self.fc = nn.Sequential(
        #     nn.Conv2d(self.channels, self.num_classes, kernel_size=1,
        #               stride=1, padding=0, bias=True)
        # )

        up_factor = 2 ** self.first_level
        if up_factor > 1:
            up = nn.ConvTranspose2d(self.num_classes, self.num_classes, up_factor * 2,
                                    stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=self.num_classes,
                                    bias=False)
        else:
            up = -1 # TODO unimplemented

        self.up = up
        # self.softmax = nn.LogSoftmax(dim=1)
        # self.init_weights() # TODO other configs do not use this function explictly

    def init_weights(self):
        fill_up_weights(self.up)
        self.up.weight.requires_grad = False
        n = self.conv_seg.kernel_size[0] * self.conv_seg.kernel_size[1] * self.conv_seg.out_channels
        self.conv_seg.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, inputs):
        x = self.cls_seg(inputs)
        y = self.up(x)
        # y = self.softmax(y)
        return y
    
 #   @force_fp32(apply_to=('seg_logit', ))
  #  def losses(self, seg_logit, seg_label):
  #v      """Compute segmentation loss."""
        #vloss = dict()
       # seg_logit = resize(
           # input=seg_logit,
          #  size=seg_label.shape[2:],
         #   mode='bilinear',
        #    align_corners=self.align_corners)
       # if self.sampler is not None:
       #     seg_weight = self.sampler.sample(seg_logit, seg_label)
       # else:
        #    seg_weight = None
       # seg_label = seg_label.squeeze(1)
#        class_weight = torch.FloatTensor([1] + [self.fg_weight] * (self.num_classes - 1)).to(seg_logit.device)
        # print('\npin', self.num_classes, seg_label.max(), 'pin\n')
       # loss['loss_seg'] = self.loss_decode(
       #     seg_logit,
      #      seg_label,
     #       weight=seg_weight,
 #           class_weight=class_weight,
    #        ignore_index=self.ignore_index)
   #     loss['acc_seg'] = accuracy(seg_logit, seg_label)
  #      return loss
