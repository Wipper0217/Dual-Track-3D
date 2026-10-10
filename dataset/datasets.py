from __future__ import absolute_import, division

import numpy as np
import os
import cv2
from torch.utils.data import Dataset
import torch

__all__ = ['Pair', 'PrePair']


class Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(Pair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]
        img_file = self.seqs[index]

        z_file, x_file, box_z, box_x = self._sample_pair(img_file)

        z = cv2.imread(z_file)
        x = cv2.imread(x_file)

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)

        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, img_file):
        z_file = img_file
        #print("z file=", z_file)
        anno_dirs = os.path.abspath(os.path.join(z_file, os.pardir, os.pardir, 'Annotation'))
        #print("anno_dirs=", anno_dirs)
        anno_files = [os.path.join(anno_dirs, file) for file in os.listdir(anno_dirs)
                      if file.endswith('.txt') and not file.startswith('wh')]

        anno_index = np.random.choice([i for i in range(len(anno_files))])
        anno_file = anno_files[anno_index]

        wh_file = os.path.join(os.path.split(anno_file)[0], 'wh_manual.txt')
        whs = np.loadtxt(wh_file)
        if whs.ndim == 2:
            wh = whs[anno_index]
        else:
            wh = whs

        anno = np.loadtxt(anno_file)
        frames = anno[:, 0]
        z_num = int(os.path.split(z_file)[-1].replace('.png', ''))
        #print("z_num=", z_num)
        index_z = np.where(frames == z_num)
        if len(frames[index_z]) == 0:
            z_num = np.random.choice(frames)
            index_z = np.where(frames == z_num)
        center_z = anno[index_z][0, 1:]
        box_z = np.append(center_z, wh)
        # convert to ltwh
        box_z[:2] = box_z[:2] - box_z[2:] / 2

        np.delete(frames, index_z)
        x_num = np.random.choice(frames)
        files = os.path.split(z_file)[0]
        x_file = os.path.join(files, os.listdir(files)[int(x_num) - 1])
        z_file = os.path.join(files, os.listdir(files)[int(z_num) - 1])
        #print("z_file={}, x_file={}".format(z_file, x_file))
        index_x = np.where(frames == x_num)
        center_x = anno[index_x][0, 1:]
        box_x = np.append(center_x, wh)
        # convert to ltwh
        box_x[:2] = box_x[:2] - box_x[2:] / 2

        return z_file, x_file, box_z, box_x


class CascadedPair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(CascadedPair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]
        img_file = self.seqs[index]

        z_file, x_file, box_z, box_x = self._sample_pair(img_file)

        z = cv2.imread(z_file)
        x = cv2.imread(x_file)

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)

        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, img_file):
        z_file = img_file
        #print("z file=", z_file)
        anno_dirs = os.path.abspath(os.path.join(z_file, os.pardir, os.pardir, 'Annotation'))
        #print("anno_dirs=", anno_dirs)
        anno_files = [os.path.join(anno_dirs, file) for file in os.listdir(anno_dirs)
                      if file.endswith('.txt') and not file.startswith('wh')]

        anno_index = np.random.choice([i for i in range(len(anno_files))])
        anno_file = anno_files[anno_index]

        wh_file = os.path.join(os.path.split(anno_file)[0], 'wh_manual.txt')
        whs = np.loadtxt(wh_file)
        if whs.ndim == 2:
            wh = whs[anno_index]
        else:
            wh = whs

        anno = np.loadtxt(anno_file)
        frames = anno[:, 0]
        z_num = int(os.path.split(z_file)[-1].replace('.png', ''))
        #print("z_num=", z_num)
        index_z = np.where(frames == z_num)
        if len(frames[index_z]) == 0:
            z_num = np.random.choice(frames)
            index_z = np.where(frames == z_num)
        center_z = anno[index_z][0, 1:]
        box_z = np.append(center_z, wh)
        # convert to ltwh
        box_z[:2] = box_z[:2] - box_z[2:] / 2

        np.delete(frames, index_z)
        x_num = np.random.choice(frames)
        files = os.path.split(z_file)[0]
        x_file = os.path.join(files, os.listdir(files)[int(x_num) - 1])
        z_file = os.path.join(files, os.listdir(files)[int(z_num) - 1])
        #print("z_file={}, x_file={}".format(z_file, x_file))
        index_x = np.where(frames == x_num)
        center_x = anno[index_x][0, 1:]
        box_x = np.append(center_x, wh)
        # convert to ltwh
        box_x[:2] = box_x[:2] - box_x[2:] / 2

        return z_file, x_file, box_z, box_x


class PrePair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(PrePair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]
        img_file = self.seqs[index]
        img_file, aug_img_file, origin_corner, corner = self._sample_pair(img_file)

        z = cv2.imread(img_file)
        x = cv2.imread(aug_img_file)
        item = (z, x, (origin_corner[0], origin_corner[1], 18, 18),
                (corner[0], corner[1], 18, 18))
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, img_file):
        aug_img_file = os.path.abspath(
            os.path.join(img_file, os.pardir, os.pardir, 'Augmentation', os.path.split(img_file)[-1]))
        origin_corners = np.loadtxt(aug_img_file.replace('.png', '_o_.txt'))
        corners = np.loadtxt(aug_img_file.replace('.png', '.txt'))
        index = np.random.randint(corners.shape[0])
        return img_file, aug_img_file, origin_corners[index], corners[index]
