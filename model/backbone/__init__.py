# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from model.backbone.swin_transformer import build_swint



def build_backbone(name, **kwargs):
    return build_swint()

