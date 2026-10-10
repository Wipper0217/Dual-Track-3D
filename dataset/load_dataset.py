import os
import numpy as np
from itertools import chain

__all__ = ['LoadDataSet', 'LoadPreDataSet']


class LoadDataSet:
    def __init__(self, root_dir):
        series_name_list = os.listdir(root_dir)
        img_dirs = [os.path.join(root_dir, series, 'Data') for series in series_name_list]
        total_img_anno_files = []
        for img_dir in img_dirs:
            series_name = os.path.abspath(os.path.join(img_dir, os.pardir))
            anno_file = os.path.join(series_name, 'Annotation', os.path.split(series_name)[-1]+'_1.txt')
            frames = (np.loadtxt(anno_file)[:, 0]-1).astype(int).tolist()
            img_files = os.listdir(img_dir)
            total_img_anno_files.append([os.path.join(img_dir, file) for file in map(img_files.__getitem__, frames)])
        self.total_img_anno_files = list(chain.from_iterable(total_img_anno_files))

    def __getitem__(self, index):
        return self.total_img_anno_files[index]

    def __len__(self):
        return len(self.total_img_anno_files)


class LoadPreDataSet:

    def __init__(self, root_dir):
        series_name_list = os.listdir(root_dir)
        img_dirs = [os.path.join(root_dir, series, 'Data') for series in series_name_list]
        img_files = [[os.path.join(img_dir, dir)for dir in os.listdir(img_dir)] for img_dir in img_dirs]
        self.total_files = list(chain.from_iterable(img_files))

    def __getitem__(self, index):
        # img_file = self.total_files[index]
        # aug_img_file = os.path.abspath(os.path.join(img_file, os.pardir, os.pardir, 'Augmentation', os.path.split(img_file)[-1]))
        return self.total_files[index]

    def __len__(self):
        return len(self.total_files)
