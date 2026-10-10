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
class SeriesDataset(object):
    #def __init__(self, root, name, frames, normalize, in_channels, channels_24_direction=None):
    def __init__(self, root, name, frames, normalize, crop_function, img_size=24):
        super(SeriesDataset, self).__init__()
        self.root = os.path.join(root, name, 'Data')

        self.name = name
        self.frames = frames
        self.num = len(frames)
        self.normalize = normalize

        self.crop_function = crop_function

        self.img_size = img_size

        # self.in_channels = in_channels
        # if in_channels == 24:
        #     if channels_24_direction is None:
        #         self.channels_24_direction = 'z'
        #         self.crop_function = crop_24channels_z
        #     else:
        #         self.channels_24_direction = channels_24_direction
        #         if self.channels_24_direction == 'x':
        #             self.crop_function = crop_24channels_x
        #         elif self.channels_24_direction == 'y':
        #             self.crop_function = crop_24channels_y
        #         else:
        #             self.crop_function = crop_24channels_z
        # elif in_channels == 9:
        #     self.crop_function = crop_9channels
        # elif in_channels == 3:
        #     self.crop_function = crop_3channels
        # else:
        #     raise ValueError("in_channels must be 3, 9 or 24")

    # def init_dateset(self, json_path):
    #     with open(json_path, 'r') as f:
    #         meta_data = json.load(f)
    #     self.labels = meta_data
    #     self.videos = list(self.labels.keys())
    #     self.num = len(self.labels)

    # def get_train_data(self, index):
    #     selected_series = random.choice(self.num)
    #     return self.labels



    def get_random_frame(self, method='2d'):
        #size = self.labels[series]['size']
        frames = self.frames

        selected_frame_index = random.choice(frames)
        data_files = sorted(glob(os.path.join(self.root, '*.mha')))
        data = sitk.ReadImage(data_files[selected_frame_index-1])
        data = sitk.GetArrayFromImage(data)
        return self.random_crop(data)




    def random_crop(self, data):
        crop = self.crop_function(data, self.img_size)
        if self.normalize:
            crop_normalized = crop / 255.0
            return crop_normalized
        return crop

    def test(self):
        for series in self.labels.keys():
            print(series)
        print("total  series: {}".format(len(self.labels)))

    # def isTrainData(self):
    #
    # def isValidData(self):


class Dataset3D(Dataset):
    #def __init__(self, root, normalize, in_channels, channels_24_direction=None):
    def __init__(self, root, normalize, crop_function, img_size=24):
        super(Dataset3D, self).__init__()
        self.root = root
        self.num = cfg.DATASET.NUM
        self.normalize = normalize
        #self.in_channels = in_channels
        #self.channels_24_direction = channels_24_direction
        self.crop_function = crop_function
        self.all_dataset = self._build_dataset(cfg.DATASET.JSON_PATH, img_size)
        self.pick = self.shuffle()


    def _build_dataset(self, json_path, img_size):

        series = []
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        self.labels = meta_data
        series_list = self.labels.keys()
        for series_name in series_list:
            series_dataset = SeriesDataset(self.root,
                                           series_name,
                                           self.labels[series_name],
                                           self.normalize,
                                           self.crop_function,
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
    dataset = Dataset3D(
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
    dataset = Dataset3D(
        r'D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D',
        3,

    )
    print(len(dataset))
    print(dataset[0].shape)
    dataset3d_test()
