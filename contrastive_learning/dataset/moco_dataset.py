import logging
import os
import random

import cv2
import numpy as np
from glob import glob

import torch

from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, RandSpatialCropd, RandFlipd, RandAffined,
    RandGaussianSmoothd, RandGaussianNoised, RandAdjustContrastd,
    RandShiftIntensityd, ScaleIntensityd, EnsureTyped
)
import re
import SimpleITK as sitk

from contrastive_learning.dataset.augment import Augmentation
from contrastive_learning.dataset.crop_utils import random_crop_numpy
from contrastive_learning.moco.loader import TwoCropsTransform
from core.config_moco import cfg
from model.builder.AutoEncoder import AutoEncoder
logger = logging.getLogger("moco dataset")

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MoCoSeriesDataset(object):
    def __init__(self,
                 root, name, frames, normalize, labels,
                 random_range, img_size):
        super(MoCoSeriesDataset, self).__init__()
        self.root = os.path.join(root, name, 'Data')

        self.name = name
        self.frames = frames
        self.num = len(frames)
        self.normalize = normalize

        self.labels = labels
        self.random_range = random_range
        self.img_size = img_size




    def get_random_frame(self, method='2d'):
        #size = self.labels[series]['size']
        frames = self.frames

        selected_frame_index = random.choice(frames)
        data_files = sorted(glob(os.path.join(self.root, '*.mha')))
        data = sitk.ReadImage(data_files[selected_frame_index-1])
        data = sitk.GetArrayFromImage(data)
        return self._random_crop(data)




    def _random_crop(self, data):
        # target = random.choice(self.labels)
        # center = target['pos']
        # size = target['size']
        center = random.choice(self.labels)
        size = [20, 20, 20]
        crop = random_crop_numpy(data, center, size, self.img_size, self.random_range)

        # if self.normalize:
        #     crop_normalized = crop / 255.0
        #     return crop_normalized
        return crop


class MoCoDataset3D(Dataset):
    def __init__(self, root, normalize, img_size=32):
        super(MoCoDataset3D, self).__init__()
        self.root = root
        self.num = cfg.DATASET.NUM
        self.normalize = normalize
        self.all_dataset = self._build_dataset(cfg.DATASET.JSON_PATH, cfg.DATASET.LABEL_PATH, cfg.DATASET.RANDOM_RANGE, img_size)
        self.pick = self.shuffle()
        self.augmentation = Compose([
            # normalize to [0, 1]
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            # 50% probability of random flip
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[1]),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[2]),

            # random affine (rotation + scaling, no translation)
            RandAffined(
                keys=["image"],
                prob=0.8,
                rotate_range=(0.1, 0.1, 0.1),  # x,y,z axis rotation in radians ±0.1
                scale_range=(0.1, 0.1, 0.1),  # x,y,z scaling ±10%
                mode="bilinear"
            ),
        ])
        self.two_crops = TwoCropsTransform(self.augmentation)

    def _build_dataset(self, json_path, label_path, random_range, img_size):

        series = []
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.meta_data = meta_data

        series_list = self.labels.keys()
        for series_name in series_list:
            series_dataset = MoCoSeriesDataset(self.root,
                                           series_name,
                                           self.meta_data[series_name],
                                           self.normalize,
                                           self.labels[series_name],
                                           random_range,
                                           img_size)
            series.append(series_dataset)
        return series

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for series_dataset in self.all_dataset:
                p.append(series_dataset)
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        return pick[:self.num]



    def __len__(self):
        return self.num

    def __getitem__(self, index):
        selected_series = self.pick[index]
        patch = selected_series.get_random_frame()
        data_dict = {
            "image": patch
        }
        # v = np.max(patch)
        # print(v)
        # if v < 1.0:
        #     print(selected_series.name)

        im_q, im_k = self.two_crops(data_dict)
        #print(torch.max(im_q['image']), torch.max(im_k['image']))
        #return [patch, patch]
        return [im_q['image'], im_k['image']]




