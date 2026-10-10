import numpy as np
import torch
from torch import nn
import math

from mmcv.cnn import ConvModule
from mmengine.model import normal_init, xavier_init
from mmengine.model import BaseModule
from mmdet.models.backbones.resnet import Bottleneck


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_fm_center_torch(score_size, score_offset, total_stride, device='cpu'):
    fm_height = score_size
    fm_width = score_size
    y = torch.arange(0, fm_height, dtype=torch.float, requires_grad=False, device=device)
    x = torch.arange(0, fm_width, dtype=torch.float, requires_grad=False, device=device)
    yy, xx = torch.meshgrid(y, x)

    xx = xx.unsqueeze(0).unsqueeze(-1)
    yy = yy.unsqueeze(0).unsqueeze(-1)
    fm_center = score_offset + torch.cat([xx, yy], dim=-1) * total_stride
    fm_center = fm_center.reshape(1, -1, 2)

    return fm_center


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


class BasicResBlock(BaseModule):
    """Basic residual block.

        This block is a little different from the block in the ResNet backbone.
        The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

        Args:
            in_channels (int): Channels of the input feature map.
            out_channels (int): Channels of the output feature map.
            conv_cfg (dict): The config dict for convolution layers.
            norm_cfg (dict): The config dict for normalization layers.
            init_cfg (dict or list[dict], optional): Initialization config dict.
                Default: None
    """

    def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=None, init_cfg=None):
        super(BasicResBlock, self).__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


class MyHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MyHead, self).__init__()
        num_classes = 1 #.TRAIN.NUM_CLASSES

        self.score_size = cfg.TRAIN.OUTPUT_SIZE
        self.total_stride = cfg.TRACK.STRIDE
        self.score_offset = cfg.TRAIN.OFFSET
        self.in_channels = in_channels
        self.bi = torch.nn.Parameter(torch.tensor(0.))
        self.si = torch.nn.Parameter(torch.tensor(1.))
        # ctr = get_xy_ctr_np(self.score_size, self.score_offset,
        #                     self.total_stride)
        # self.fm_ctr = ctr
        # self.fm_ctr.require_grad = False

        fm_ctr = get_fm_center_torch(self.score_size, self.score_offset, self.total_stride)
        self.register_buffer('fm_ctr', fm_ctr.clone().detach().requires_grad_(False))
        cls_tower = []
        bbox_tower = []
        mid_channels = 1024
        for _ in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(Bottleneck(inplanes=mid_channels,
                                        planes=mid_channels // 4,
                                        conv_cfg=None,
                                        norm_cfg=dict(type='BN')))
            bbox_tower.append(Bottleneck(inplanes=mid_channels,
                                         planes=mid_channels // 4,
                                         conv_cfg=None,
                                         norm_cfg=dict(type='BN')))
        self.cls_tower = nn.Sequential(
            BasicResBlock(self.in_channels, mid_channels),
            *cls_tower
        )
        self.bbox_tower = nn.Sequential(
            BasicResBlock(self.in_channels, mid_channels),
            *bbox_tower
        )
        self.cls_logits = ConvModule(in_channels=mid_channels,
                                     out_channels=1,
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN'),
                                     act_cfg=None)
        self.bbox_pred = ConvModule(in_channels=mid_channels,
                                    out_channels=4,
                                    kernel_size=1,
                                    norm_cfg=dict(type='BN'),
                                    act_cfg=None)
        self.centerness = ConvModule(in_channels=mid_channels,
                                     out_channels=1,
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN'),
                                     act_cfg=None)

    def init_weights(self):
        modules = [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness]
        for m in modules:
            for ll in m.modules():
                if isinstance(ll, nn.Conv2d):
                    normal_init(ll, std=0.01)

    def forward(self, x):

        cls_tower = self.cls_tower(x)
        cls_score = self.cls_logits(cls_tower)

        cls_score = cls_score.permute(0, 2, 3, 1)                   # (N, H, W, C)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)    # (N, HW, C)

        ctr_score = self.centerness(cls_tower)
        ctr_score = ctr_score.permute(0, 2, 3, 1)                   # (N, H, W, C)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)    # (N, HW, C)
        offsets = self.bbox_pred(self.bbox_tower(x))
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride

        bbox = self.offset2bbox(offsets)
        return cls_score, ctr_score, bbox

    def offset2bbox(self, offsets):
        bbox = get_box(self.fm_ctr, offsets)
        return bbox


if __name__ == '__main__':
    import argparse
    from pysot.core.config import cfg

    parser = argparse.ArgumentParser(description='my head track_test')
    parser.add_argument('--config', type=str,
                        default='D:\pycharmProject\pythonProject1\coding\MySiamCAR\experiments\siamcar_r50\config.yaml',
                        help='config file')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    head = MyHead(cfg, in_channels=256)
    fm = torch.rand(1, 256, 31, 31)
    cls_score, ctr_score, bbox = head(fm)
    print(cls_score.shape)
    print(ctr_score.shape)
    print(bbox.shape)