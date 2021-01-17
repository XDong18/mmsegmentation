import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import NECKS
# from ..model_utils import ConvModule
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.cnn import ConvModule, xavier_init
from mmcv.utils.parrots_wrapper import _BatchNorm


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


class IDAUp(nn.Module):

    def __init__(self,
                 node_kernel,
                 out_dim,
                 channels,
                 up_factors,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        c,
                        out_dim,
                        1,
                        stride=1,
                        bias=False),
                    build_norm_layer(norm_cfg, out_dim)[1],
                    nn.ReLU(inplace=True))

            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim,
                    out_dim,
                    f * 2,
                    stride=f,
                    padding=f // 2,
                    output_padding=0,
                    groups=out_dim,
                    bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                build_conv_layer(
                        conv_cfg,
                        out_dim * 2,
                        out_dim,
                        node_kernel,
                        stride=1,
                        padding=node_kernel // 2,
                        bias=False),
                build_norm_layer(norm_cfg, out_dim)[1],
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            # classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


@NECKS.register_module
class DLAUp(nn.Module):
    def __init__(self,
                 channels,
                 scales=(1, 2, 4, 8), # (1,2,4,8,16)
                 in_channels=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(
                    3,
                    channels[j],
                    in_channels[j:],
                    scales[j:] // scales[j],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, layers):
        layers = list(layers)
        layers = layers[6 - len(self.channels):] 
        assert len(layers) > 1
        # ms_feat = [layers[-1]]
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])  # y : aggregation nodes
            layers[-i - 1:] = y
            # ms_feat.append(x)
        # ms_feat = ms_feat[::-1]
        # ms_feat[-1] = self.last_conv(ms_feat[-1])
        # if self.num_outs > len(ms_feat):
            # ms_feat.append(F.max_pool2d(ms_feat[-1], 1, stride=2))
        return x  # x