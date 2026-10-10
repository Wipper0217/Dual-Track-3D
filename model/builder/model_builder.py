# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from model.backbone import build_backbone
from model.head.my_head import MyHead
from model.neck import get_adjust
from model.neck.global_attention_module import GlobalAttentionModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



class GAMModelBuilder(nn.Module):
    def __init__(self):
        super(GAMModelBuilder, self).__init__()
        self.backbone = build_backbone(cfg.BACKBONE.TYPE,
                                       **cfg.BACKBONE.KWARGS)
        self.adjust = get_adjust(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        self.neck = GlobalAttentionModule(mid_channels_model=256,
                                          dim_feedforward=2048,
                                          num_heads=8,
                                          num_layers=2,
                                          f_z_size=8,
                                          f_x_size=16,
                                          dropout=0.1,
                                          activation='relu')
        self.head = MyHead(cfg, 256)

        #self.loss_evaluator = make_loss_evaluator(cfg)

    def template(self, z):

        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.adjust(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.adjust(xf)
        features = self.neck(self.zf, xf)
        cls, ctr, box = self.head(features)

        cls = cls.squeeze()
        ctr = ctr.squeeze()
        box = box.squeeze()


        cls_score = torch.sigmoid(cls)
        ctr_score = torch.sigmoid(ctr)
        score = cls_score * ctr_score
        print(cls_score.shape, ctr_score.shape, box.shape, score.shape)
        outputs = {
            'cls_score': cls_score,
            'ctr_score': ctr_score,
            'box': box,
            'score': score
        }
        return outputs

    def forward(self, train_data):

        for key, value in train_data.items():
            value = value.to(device)
            train_data[key] = value

        template = train_data['template']
        search = train_data['search']

        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.adjust(zf)
            xf = self.adjust(xf)
        features = self.neck(zf, xf)
        cls, ctr, box = self.head(features)

        pred_data = dict(
            cls_pred=cls,
            ctr_pred=ctr,
            box_pred=box
        )

        cls_loss, ctr_loss, reg_loss = self.loss_evaluator(pred_data, train_data)
        total_loss = cls_loss + ctr_loss + reg_loss
        output = dict(
            cls_loss=cls_loss,
            ctr_loss=ctr_loss,
            reg_loss=reg_loss,
            total_loss=total_loss
        )
        return output



