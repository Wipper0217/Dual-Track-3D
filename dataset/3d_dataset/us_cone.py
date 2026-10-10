import json

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import os
from glob import glob

def exclude_labeled_data(path):
    series = os.listdir(path)
    unlabeled = {}
    for serie in series:
        serie_path = os.path.join(path, serie)
        serie_name = os.path.basename(serie_path)
        print("{} start!".format(serie_name))
        data_path = os.path.join(serie_path, 'Data')
        anno_path = os.path.join(serie_path, 'Annotation')
        data = sorted(glob(os.path.join(data_path, '*.mha')))
        anno = sorted(glob(os.path.join(anno_path, '*.txt')))

        union_anno_frames = []
        for i in range(len(anno)):
            #object = i+1
            anno_frames = np.loadtxt(anno[i])[:, 0].tolist()

            anno_frames = [int(j) for j in anno_frames]
            union_anno_frames.append(anno_frames)
        anno_frames = sorted(set().union(*union_anno_frames))
        cur_anno_index = 0
        unanno_frames = []
        for i in range(len(data)):
            cur_frame_index = i + 1
            anno_frame_index = anno_frames[cur_anno_index]
            if cur_frame_index == anno_frame_index:
                if cur_anno_index + 1 < len(anno_frames):
                    cur_anno_index += 1
            else:
                unanno_frames.append(cur_frame_index)
        if len(unanno_frames) > 0:
            unlabeled[serie] = unanno_frames
    with open('train.json', 'w', encoding='utf-8') as f:
        json.dump(unlabeled, f, ensure_ascii=False, indent=4, sort_keys=True)


def using_unlabeled_data(path):
    series = os.listdir(path)
    unlabeled = {}
    for serie in series:
        serie_path = os.path.join(path, serie)
        serie_name = os.path.basename(serie_path)
        print("{} start!".format(serie_name))
        data_path = os.path.join(serie_path, 'Data')
        anno_path = os.path.join(serie_path, 'Annotation')
        data = sorted(glob(os.path.join(data_path, '*.mha')))
        anno = sorted(glob(os.path.join(anno_path, '*.txt')))
        frames = []
        for i in range(len(data)):
            frames.append(i+1)

        unlabeled[serie] = frames
    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(unlabeled, f, ensure_ascii=False, indent=4, sort_keys=True)


def get_anno(path):
    series = os.listdir(path)
    labels = {}
    for serie in series:
        serie_path = os.path.join(path, serie)
        serie_name = os.path.basename(serie_path)
        print("{} start!".format(serie_name))
        data_path = os.path.join(serie_path, 'Data')
        anno_path = os.path.join(serie_path, 'Annotation')
        data = sorted(glob(os.path.join(data_path, '*.mha')))
        anno = sorted(glob(os.path.join(anno_path, '*.txt')))
        labels[serie_name] = []
        for i in range(len(anno)):
            target_name = os.path.basename(anno[i]).split('.')[0]
            label = np.loadtxt(anno[i])[1:].tolist()

            labels[serie_name].append(label)
        print(labels[serie_name])

    with open('test_label.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == '__main__':

    #exclude_labeled_data(r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D")
    #using_unlabeled_data(r"D:\DataSet\MICCAI_CLUST_2015_TestSet_3D\MICCAI_CLUST_2015_TestSet_3D")
    get_anno(r"D:\DataSet\MICCAI_CLUST_2015_TestSet_3D\MICCAI_CLUST_2015_TestSet_3D")

    # path = r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D\EMC-01\Data\001.mha"
    # image = sitk.ReadImage(path)
    # print(image.GetSize())
    #
    # image_array = sitk.GetArrayFromImage(image) # z, y, x
    # d, h, w = image_array.shape
    # print(d, h, w)
    # cz, cy, cx = d//2, h//2, w//2
    # print(cz, cy, cx)
    # length = 10
    #
    # for i in range(-length, length+1):
    #     print(i)
    #     layer_xy = image_array[cz+i, :, :]
    #     cv2.imshow("layer xy", layer_xy)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     layer_zx = image_array[:, cy+i, :]
    #     cv2.imshow("layer zx", layer_zx)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     layer_zy = image_array[:, :, cx+i]
    #     cv2.imshow("layer zy", layer_zy)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
