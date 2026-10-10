from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.xcorr import DepthwiseXCorr

class FCppHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FCppHead, self).__init__()
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        # (1) Feature enhancement network
        # Group normalization, which avoids the influence of batch size on the model; it splits the channel dimension into groups and then normalizes within each group https://blog.csdn.net/duanshao/article/details/80055887
        # cls branch, 4 layers: [convolution -- group normalization -- activation], [convolution -- group normalization -- activation], [convolution -- group normalization -- activation], [convolution -- group normalization -- activation]
        # reg branch, 4 layers: [convolution -- group normalization -- activation], [convolution -- group normalization -- activation], [convolution -- group normalization -- activation], [convolution -- group normalization -- activation]
        for i in range(cfg.TRAIN.NUM_CONVS):  #
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # (2) Classification branch output  input:[256] --> output:[2]
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        # (3) Regression branch output  input:[256] --> output:[4]
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        # (4) Center branch output  input:[256] --> output:[1]
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        out_channels = in_channels
        self.cls_dw = DepthwiseXCorr(in_channels, out_channels)
        self.bbox_dw = DepthwiseXCorr(in_channels, out_channels)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, xf, zf):
        x_cls = self.cls_dw(xf, zf)     #[B, 256, 25, 25] --> [B, 256, 25, 25]
        x_reg = self.bbox_dw(xf, zf)    # [B, 256, 25, 25] --> [B, 2, 25, 25]
        cls_tower = self.cls_tower(x_cls)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_tower = self.bbox_tower(x_reg)
        bbox_reg = self.bbox_pred(bbox_tower)
        bbox_reg = torch.exp(bbox_reg)

        #print("shape :logits={}, bbox_reg={}, centerness={}".format(logits.shape, bbox_reg.shape, bbox_tower.shape))

        return logits, bbox_reg, centerness