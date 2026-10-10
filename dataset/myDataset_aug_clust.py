# Copyright (c) Yifan. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')

from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        self.num = 0
        for name in cfg.DATASET.NAMES:
            meta_data = {}
            subdata_cfg = getattr(cfg.DATASET, name)
            data_dir = subdata_cfg.ROOT
            series_path = sorted(glob(os.path.join(data_dir, 'sub[0-9]', '*-*', 'Augmentation')))
            for ser_path in series_path:
                images_path = sorted(glob(os.path.join(ser_path, '*.png')))
                meta_data[ser_path] = len(images_path)

            self.all_dataset.append(meta_data)

        # data augmentation
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP
        )

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for meta_data in self.all_dataset:
                sub_p = list(range(len(meta_data.keys())))
                np.random.shuffle(sub_p)
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            return dataset, index

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        video_name = list(dataset.keys())[index]
        video_frame = dataset[video_name]
        template_frame = np.random.randint(0, video_frame)

        img_files = sorted(glob(os.path.join(video_name, '*.png')))
        search_path = img_files[template_frame]
        template_path = search_path.replace('Augmentation', 'Data')
        labels = np.loadtxt(search_path.replace('.png', '_o_.txt'))
        label_index = np.random.randint(0, labels.shape[0])
        template_center = labels[label_index]
        search_center = np.loadtxt(search_path.replace('.png', '.txt'))[label_index]

        # get image
        template_image = cv2.imread(template_path)
        search_image = cv2.imread(search_path)

        s_z = 20
        scale_z = 127 / s_z
        d_search = (255 - 127) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        a = (255-1) / s_x
        b = (255-1) / s_x

        template_mapping = np.array([[a, 0, -a * (template_center[0]-4.5)],
                                     [0, b, -b * (template_center[1]-4.5)]]).astype(np.float)
        search_mapping = np.array([[a, 0, -a * (search_center[0]-4.5)],
                                   [0, b, -b * (search_center[1]-4.5)]]).astype(np.float)
        template_crop = cv2.warpAffine(template_image, template_mapping, (255, 255), borderMode=cv2.BORDER_CONSTANT, borderValue=np.mean(template_image, axis=(0, 1)))
        search_crop = cv2.warpAffine(search_image, search_mapping, (255, 255), borderMode=cv2.BORDER_CONSTANT, borderValue=np.mean(search_image, axis=(0, 1)))


        # get bounding box
        template_box = self._get_bbox(template_crop, [template_center[0]-5, template_center[1]-5,
                                                      template_center[0]+5, template_center[1]+5])
        search_box = self._get_bbox(search_crop, [search_center[0]-5, search_center[1]-5,
                                                  search_center[0]+5, search_center[1]+5])


        # augmentation
        template, _ = self.template_aug(template_crop,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE)

        search, bbox = self.search_aug(search_crop,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE)


        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'bbox': np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        }

