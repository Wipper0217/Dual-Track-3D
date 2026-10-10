# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "moco"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()
__C.TRAIN.LR = 0.0001


__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.SAVE_EPOCH = 10

__C.TRAIN.BATCH_SIZE = 2

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001


__C.TRAIN.PRINT_FREQ = 10

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 1.0

__C.TRAIN.BASE_LR = 0.001
# __C.TRAIN.BACKBONE.LR = 0.0005
# __C.TRAIN.HEAD.LR = 0.0004

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'
__C.TRAIN.LR.LR = 0.01
__C.TRAIN.LR.COS = True

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = False

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)



__C.TRAIN.PRIOR_PROB = 0.01




__C.TRAIN.LR_SCHEDULER = CN()
__C.TRAIN.LR_SCHEDULER.TYPE = 'Exponential'
__C.TRAIN.LR_SCHEDULER.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
# __C.DATASET.TEMPLATE = CN()


__C.DATASET.IMG_SIZE = 24
__C.DATASET.NUM = 100
__C.DATASET.ROOT = r'e:\1DATA\CLUST\CLUST_3D\clust_testset_3d'
__C.DATASET.JSON_PATH = r'e:\1DATA\CLUST\CLUST_3D\clust_testset_3d\test.json'
__C.DATASET.LABEL_PATH = r"e:\1DATA\CLUST\CLUST_3D\clust_testset_3d\test_label.json"
__C.DATASET.RANDOM_RANGE = 12
__C.DATASET.NORMALIZE = True


__C.ENCODER = CN()
__C.ENCODER.TYPE = 'resnet34'
__C.ENCODER.KWARGS = CN(new_allowed=True)

__C.MOCO = CN()
__C.MOCO.KWARGS = CN(new_allowed=True)
__C.TRACKER = CN()
__C.TRACKER.LAMBDA_SPATIAL = 0.0
__C.TRACKER.LAMBDA_AUTOENCODER = 1.0
__C.TRACKER.LAMBDA_ENCODER = 1.0
__C.TRACKER.MODE = 'normal'


# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'resnet50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 0

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"


