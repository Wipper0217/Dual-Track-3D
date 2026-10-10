# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()
__C.TRAIN.LR = 0.0001
# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 31

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

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 1.0

__C.TRAIN.BASE_LR = 0.001
# __C.TRAIN.BACKBONE.LR = 0.0005
# __C.TRAIN.HEAD.LR = 0.0004

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = False

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

__C.TRAIN.FOCAL_LOSS_ALPHA = 0.5
__C.TRAIN.FOCAL_LOSS_GAMMA = 0
__C.TRAIN.LOSS_WEIGHT = 1.0

__C.TRAIN.OFFSET = 31

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
__C.DATASET.ROOT = r'D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D'
__C.DATASET.JSON_PATH = r'D:\pycharmProject\pythonProject1\coding\3dTracking\dataset\3d_dataset\output.json'
__C.DATASET.LABEL_PATH = r"D:\pycharmProject\pythonProject1\coding\3dTracking\dataset\3d_dataset\test_label.json"
__C.DATASET.RANDOM_RANGE = 12
__C.DATASET.NORMALIZE = True
__C.DATASET.METHOD = '2d'
__C.DATASET.IN_CHANNELS = 3
__C.DATASET.CHANNELS_24_DIRECTION = 'x'




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

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44
__C.TRACK.XY_WINDOW_INFLUENCE = 0.44
__C.TRACK.XZ_WINDOW_INFLUENCE = 0.44
__C.TRACK.YZ_WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.INIT_CONTEXT_AMOUNT = 0.4
__C.TRACK.CONTEXT_AMOUNT = 0.2

__C.TRACK.STRIDE = 8


__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.CLUST = [0.05, 0.01, 0.6]

# __C.HP_SEARCH.CLUST.CIL = []
# __C.HP_SEARCH.CLUST.ETH = []
# __C.HP_SEARCH.CLUST.ICR = []
# __C.HP_SEARCH.CLUST.MED = []

__C.HP_SEARCH.OTB100 = [0.35, 0.2, 0.45]

__C.HP_SEARCH.GOT10K = [0.7, 0.06, 0.1]

__C.HP_SEARCH.UAV123 = [0.4, 0.2, 0.3]


__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.3]
