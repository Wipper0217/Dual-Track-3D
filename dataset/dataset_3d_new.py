import logging
import os
import random

import cv2
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import re
import SimpleITK as sitk
from dataset.crop_utils import crop_3channels, crop_9channels, crop_24channels_x, crop_24channels_y, crop_24channels_z
from core.config_3d import cfg
from model.builder.AutoEncoder import AutoEncoder
logger = logging.getLogger("global")

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SeriesDatasetNew(object):
    def __init__(self, root, name, frames, normalize, crop_function, labels, random_range, img_size=24):
        super(SeriesDatasetNew, self).__init__()
        self.root = os.path.join(root, name, 'Data')

        self.name = name
        self.frames = frames
        self.num = len(frames)
        self.normalize = normalize

        self.crop_function = crop_function
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
        center = random.choice(self.labels)
        print("series: {}, center={}".format(self.name, center))
        crop = self.crop_function(data, center, self.random_range, self.img_size)
        if self.normalize:
            crop_normalized = self._normalize(crop)
            return crop_normalized
        return crop

    def _normalize(self, img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-6:
            return (img - img_min) / (img_max - img_min)
        return img
    def test(self):
        for series in self.labels.keys():
            print(series)
        print("total  series: {}".format(len(self.labels)))

    # def isTrainData(self):
    #
    # def isValidData(self):


class Dataset3DNew(Dataset):
    #def __init__(self, root, normalize, in_channels, channels_24_direction=None):
    def __init__(self, root, normalize, crop_function, img_size=24):
        super(Dataset3DNew, self).__init__()
        self.root = root
        self.num = cfg.DATASET.NUM
        self.normalize = normalize
        #self.in_channels = in_channels
        #self.channels_24_direction = channels_24_direction
        self.crop_function = crop_function
        self.all_dataset = self._build_dataset(cfg.DATASET.JSON_PATH, cfg.DATASET.LABEL_PATH, cfg.DATASET.RANDOM_RANGE, img_size)
        self.pick = self.shuffle()


    def _build_dataset(self, json_path, label_path, random_range, img_size):

        series = []
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.meta_data = meta_data
        series_list = self.labels.keys()
        for series_name in series_list:
            series_dataset = SeriesDatasetNew(self.root,
                                           series_name,
                                           self.meta_data[series_name],
                                           self.normalize,
                                           self.crop_function,
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
        return selected_series.get_random_frame()

def dataset3d_test():
    dataset = Dataset3DNew(
        r'D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D',
        3,
    )
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = AutoEncoder(3).to(device)
    for epoch in range(1):
        for idx, data in enumerate(dataloader):
            data = data.to(device)
            print("index: {}, input={}".format(idx, data.shape))
            output = model(data)
            print("output: {}".format(output.shape))

if __name__ == '__main__':
    dataset = Dataset3DNew(
        r'D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D',
        3,

    )
    print(len(dataset))
    print(dataset[0].shape)
    dataset3d_test()
