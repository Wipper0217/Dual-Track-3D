# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.neck.neck import AdjustLayer, AdjustAllLayer, FeatureAdapter

ADJUST = {
    'AdjustLayer': AdjustLayer,
    'AdjustAllLayer': AdjustAllLayer,
    'FeatureAdapter': FeatureAdapter
    }

def get_adjust(name, **kwargs):
    return ADJUST[name](**kwargs)
