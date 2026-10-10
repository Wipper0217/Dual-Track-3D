import numpy as np
import math
import os
from glob import glob

from track_test.utils.file_util import copy_specific_files

dataset_path = r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D"

def get_gt_file(dataset_path):
    """

    :param dataset_path:
    :return:
    {
    'gt':[],
    'res': ''
    }
    """
    series = os.listdir(dataset_path)
    print(series)
    labels = {}
    for s in series:
        path = os.path.join(dataset_path, s)
        res_path = sorted(glob(os.path.join(path, "res.txt")))[0]
        anno_path = sorted(glob(os.path.join(path, "Annotation", "*.txt")))
        labels[s] = {}
        labels[s]['gt'] = anno_path
        labels[s]['res'] = res_path
    return labels


def calculate_total_error(predicts, labels):
    series = labels.keys()
    errors = []
    for s in series:
        if s not in ['EMC-02']:
            continue
        # if s not in ['SMT-03']:
        #     continue
        gt_files = labels[s]['gt']
        res_file = labels[s]['res']
        predict = predicts[s]
        error = calculate_error(s, predict, gt_files, res_file)
        errors.extend(error)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    te95th = np.percentile(errors, 95)

    print("Total mean(mm): {}, std(mm): {}, te95th(mm): {}".format(error_mean, error_std, te95th))

def calculate_sequence_error(predicts, labels, sequence):
    series = labels.keys()
    errors = []
    for s in series:
        if s != sequence:
            continue
        gt_files = labels[s]['gt']
        res_file = labels[s]['res']
        predict = predicts[s]
        error = calculate_error(s, predict, gt_files, res_file)
        errors.extend(error)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    te95th = np.percentile(errors, 95)

    print("Total mean(mm): {}, std(mm): {}, te95th(mm): {}".format(error_mean, error_std, te95th))

def get_regular_predicts(epoch=55):
    predicts = {
        "EMC-02":
            [
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_EMC-02_1_siamfc_regular.txt'.format(epoch),
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_EMC-02_2_siamfc_regular.txt'.format(epoch),
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_EMC-02_3_siamfc_regular.txt'.format(epoch),
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_EMC-02_4_siamfc_regular.txt'.format(epoch)
            ],
        "SMT-03":
            [
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_SMT-03_1_siamfc_regular.txt'.format(epoch),
                r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc_new\siamfc_3d_e{}.pth_SMT-03_2_siamfc_regular.txt'.format(epoch)
            ]
    }
    return predicts



def calculate_error(series_name, predicts, gt_files, res_file):
    errors_series = []
    target_num = len(gt_files)
    res = np.loadtxt(res_file)
    for i in range(target_num):
        gt = np.loadtxt(gt_files[i])
        gt_frames = gt[:, 0].tolist()

        predict = np.loadtxt(predicts[i])
        errors = []
        for j in range(len(gt_frames)):
            frame_idx = int(gt_frames[j]) - 1
            gt_center = gt[j, 1:]
            predict_center = predict[frame_idx]
            error = predict_center - gt_center
            error_mm = error * res
            error = np.sqrt(error_mm[0]**2 + error_mm[1]**2 + error_mm[2]**2)
            errors.append(error)
            errors_series.append(error)
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        te95th = np.percentile(errors, 95)
        print("Series: {} target-{} mean(mm): {}, std(mm): {}, te95th(mm): {}".format(series_name, i + 1, error_mean, error_std, te95th))
    return errors_series

if __name__ == '__main__':
    series = ['EMC-01', 'EMC-02', 'EMC-03', 'ICR-01', 'SMT-01', 'SMT-02', 'SMT-03', 'SMT-04']
    siamfc_epoch = 75
    predicts = {
        'EMC-02': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_EMC-02_1_siamfc.txt'.format(siamfc_epoch),
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_EMC-02_2_siamfc.txt'.format(siamfc_epoch),
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_EMC-02_3_siamfc.txt'.format(siamfc_epoch),
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_EMC-02_4_siamfc.txt'.format(siamfc_epoch)],

        'SMT-03': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_SMT-03_1_siamfc.txt'.format(siamfc_epoch),
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\siamfc\siamfc_3d_e{}.pth_SMT-03_2_siamfc.txt'.format(siamfc_epoch)],
    }
    epoch = 63
    regular_predicts = get_regular_predicts(epoch=epoch)
    labels = get_gt_file(dataset_path)
    calculate_sequence_error(predicts, labels, "EMC-02")
    calculate_sequence_error(predicts, labels, "SMT-03")

    # siamfc_folder = r"C:\Users\hello\Desktop\0_毕业论文\3d\compare\siamfc"
    # for k, v in predicts.items():
    #     print("Start copying {}".format(k))
    #     copy_specific_files(v, siamfc_folder, k)

    # calculate_total_error(regular_predicts, labels)
    # print("--------------------------regular epoch: {}--------------------------".format(epoch))
    # calculate_sequence_error(regular_predicts, labels, "EMC-02")
    # calculate_sequence_error(regular_predicts, labels, "SMT-03")
    # regular_folder = r"C:\Users\hello\Desktop\0_毕业论文\3d\compare\regular"
    # for k, v in regular_predicts.items():
    #     print("Start copying {}".format(k))
    #     copy_specific_files(v, regular_folder, k)