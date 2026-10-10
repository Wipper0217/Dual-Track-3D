# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from collections import namedtuple

import torch

Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import re
from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg
from pysot.datasets.target.make_densebox_target import make_densebox_target
from pysot.datasets.utils import normalize
logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx, mode='.png'):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.anno = anno
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}' + mode
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return cv2.imread(image_path), image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class AugSubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.anno = anno
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                                  filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.png'
        self.pick = self.shuffle()

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_search_frame = np.random.randint(0, len(frames))
        img_files = sorted(glob(os.path.join(self.root, video_name, 'Data', '*.png')))
        template_img_file = img_files[template_search_frame]
        #search_img_file = template_img_file.replace('Data', 'Augmentation')
        search_img_file = re.sub(r'\bData\b', 'Augmentation', template_img_file)
        template_img = cv2.imread(template_img_file)
        search_img = cv2.imread(search_img_file)
        template_label = np.loadtxt(search_img_file.replace('.png', '_o_.txt'))
        search_label = np.loadtxt(search_img_file.replace('.png', '.txt'))
        label_index = np.random.randint(0, template_label.shape[0])
        template_center = template_label[label_index]
        search_center = search_label[label_index]

        s_z = 20
        scale_z = 127 / s_z
        d_search = (255 - 127) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        a = (255-1) / s_x
        b = (255-1) / s_x

        template_mapping = np.array([[a, 0, -a * (template_center[0]-4.5)],
                                     [0, b, -b * (template_center[1]-4.5)]]).astype(np.float64)
        search_mapping = np.array([[a, 0, -a * (search_center[0]-4.5)],
                                   [0, b, -b * (search_center[1]-4.5)]]).astype(np.float64)
        template_crop = cv2.warpAffine(template_img, template_mapping, (255, 255), borderMode=cv2.BORDER_CONSTANT, borderValue=np.mean(template_img, axis=(0, 1)))
        search_crop = cv2.warpAffine(search_img, search_mapping, (255, 255), borderMode=cv2.BORDER_CONSTANT, borderValue=np.mean(search_img, axis=(0, 1)))

        return (template_crop, [template_center[0]-4.5, template_center[1]-4.5,
                                template_center[0]+4.5, template_center[1]+4.5]), \
               (search_crop, [search_center[0]-4.5, search_center[1]-4.5,
                              search_center[0]+4.5, search_center[1]+4.5])

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self, cfg):
        super(TrkDataset, self).__init__()
        self.cfg = cfg
        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(self.cfg.DATASET, name)
            sub_dataset = None
            if name == 'CLUST_AUG':
                sub_dataset = AugSubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            elif name == 'CLUST':
                sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start,
                    mode='.png'
                )
            else:
                sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start,
                    mode='.jpg'
                )

            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                self.cfg.DATASET.TEMPLATE.SHIFT,
                self.cfg.DATASET.TEMPLATE.SCALE,
                self.cfg.DATASET.TEMPLATE.BLUR,
                self.cfg.DATASET.TEMPLATE.FLIP
            )
        self.search_aug = Augmentation(
                self.cfg.DATASET.SEARCH.SHIFT,
                self.cfg.DATASET.SEARCH.SCALE,
                self.cfg.DATASET.SEARCH.BLUR,
                self.cfg.DATASET.SEARCH.FLIP
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, data):
        image, shape = data[0], data[1]
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

        gray = self.cfg.DATASET.GRAY and self.cfg.DATASET.GRAY > np.random.random()
        neg = self.cfg.DATASET.NEG and self.cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        if template[0] is None:
            print('error image:',template[0])

        # get bounding box
        template_box = self._get_bbox(template)
        search_box = self._get_bbox(search)


        # augmentation
        template, _ = self.template_aug(template[0],
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search[0],
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        template = normalize(template)
        search = normalize(search)
        cv2.rectangle(search, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (0,255,0), 1)
        cv2.imshow('template', template)
        cv2.imshow('search', search)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        offset = (self.cfg.TRACK.INSTANCE_SIZE - 1 - (self.cfg.TRACK.SCORE_SIZE-1)*self.cfg.TRACK.STRIDE)//2
        config = dict(
            x_size=self.cfg.TRACK.INSTANCE_SIZE,
            score_size=self.cfg.TRACK.SCORE_SIZE,
            total_stride=self.cfg.TRACK.STRIDE,
            score_offset=offset
        )
        gt_box = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2]])
        # label_cls=(score_size*score_size, 1), label_cen=(score_size*score_size, 1), label_box=(score_size*score_size, 4)
        label_cls, label_cen, label_box = make_densebox_target(gt_boxes=gt_box, config=config)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        # template = torch.from_numpy(template)
        # search = torch.from_numpy(search)
        # return {
        #         'template': template.to(device),
        #         'search': search.to(device),
        #         'bbox': torch.from_numpy(gt_box).to(device),
        #         'cls_gt': label_cls.to(device),
        #         'ctr_gt': label_cen.to(device),
        #         'box_gt': label_box.to(device)
        #         }
        return {
                'template': template,
                'search': search,
                'bbox': gt_box,
                'cls_gt': label_cls.numpy(),
                'ctr_gt': label_cen.numpy(),
                'box_gt': label_box.numpy()
                }

if __name__ == '__main__':

    dataset = TrkDataset(cfg)
    train_loader = DataLoader(dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=1,
                              pin_memory=False,
                              shuffle=True
                              )
    for idx, data in enumerate(train_loader):
        print(data.keys())
        print(data['ctr_gt'].shape)
