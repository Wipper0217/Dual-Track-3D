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
        # if s != 'EMC-02':
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

def get_best_predicts():
    predicts = {
        'EMC-01': [
            r"C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-01_1.txt"],
        'EMC-02': [
            r"C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-02_1.txt",
            r"C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-02_2.txt",
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-02_3.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-02_4.txt'],
        'EMC-03': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\EMC-03_1.txt'],
        'ICR-01': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\ICR-01_1.txt'],
        'SMT-01': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-01_1.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-01_2.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-01_3.txt'],
        'SMT-02': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-02_1.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-02_2.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-02_3.txt'],
        'SMT-03': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-03_1.txt',
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-03_2.txt'],
        'SMT-04': [
            r'C:\Users\hello\Desktop\0_毕业论文\3d\core_track\SMT-04_1.txt']
    }
    return predicts

def get_ae_predicts():
    # ae_predicts = {
    #     'EMC-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-01_1'],
    #     'EMC-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
    #     'EMC-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-03_1'],
    #     'ICR-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_ICR-01_1'],
    #     'SMT-01': [
    #         #r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_3'],
    #     'SMT-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_3'],
    #     'SMT-03': [
    #         # r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_1',
    #         # r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_2'
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_1_neighbor26.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_2_neighbor26.txt'
    #     ],
    #     'SMT-04': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    # }

    ae_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-01_1'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-03_1'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_ICR-01_1'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_3'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_3'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_2'
        ],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    }
    return ae_predicts

def get_moco_predicts():
    moco_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\moco\autoencoder_e100_EMC-01_1_neighbor_6_size_32.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_EMC-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_EMC-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_EMC-02_3_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\moco\autoencoder_e100_EMC-02_4_neighbor_6_size_32.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\moco\autoencoder_e100_EMC-03_1_neighbor_6_size_32.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_ICR-01_1_neighbor_6_size_32.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-01_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e290_SMT-01_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-01_3_neighbor_6_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-02_3_neighbor_6_size_32.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-03_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\autoencoder_e100_SMT-03_2_neighbor_6_size_32.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-04_1_neighbor_6_size_32.txt']
    }
    # 26-neighborhood
    # moco_predicts = {
    #     'EMC-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-01_1_spatial_0.0.txt'],
    #     'EMC-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_1_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_2_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_3_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_4_spatial_0.0.txt'],
    #     'EMC-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_EMC-03_1_spatial_0.0.txt'],
    #     'ICR-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_ICR-01_1_spatial_0.0.txt'],
    #     'SMT-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_SMT-01_1_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.0_size_24.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.0.txt'],
    #     'SMT-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_1_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_2_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_3_spatial_0.0.txt'],
    #     'SMT-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e155_SMT-03_1_spatial_0.0.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e210_SMT-03_2_spatial_0.0.txt'],
    #     'SMT-04': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\moco\moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.0_size64.txt']
    # }
    return moco_predicts

def get_multi_branch_n6_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-01_1_neighbor_6_size_32.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-02_3_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-02_4_neighbor_6_size_32.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-03_1_neighbor_6_size_32.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_ICR-01_1_neighbor_6_size_32.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_3_SMT-01_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_3_SMT-01_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_3_SMT-01_3_neighbor_6_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_3_SMT-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_3_neighbor_6_size_32.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_2_neighbor_6_size_32.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_2_SMT-04_1_neighbor_6_size_32.txt']
    }

    return predicts

def get_multi_branch_n6_ncc_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_EMC-01_1_neighbor_6_size_32_ncc.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_EMC-02_1_neighbor_6_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_3_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_4_neighbor_6_size_32.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-03_1_neighbor_6_size_32.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_ICR-01_1_neighbor_6_size_32_ncc.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\multi_branch_test_3_SMT-01_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-01_2_neighbor_6_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-01_3_neighbor_6_size_32_ncc.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-02_1_neighbor_6_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-02_2_neighbor_6_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e290_SMT-02_3_neighbor_6_size_32_ncc.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_2_neighbor_6_size_32.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e290_SMT-04_1_spatial_0.001_size_32.txt']
    }

    return predicts

def get_ae_n6_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-01_1_neighbor_6_size_32.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-02_3_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-02_4_neighbor_6_size_32.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_EMC-03_1_neighbor_6_size_32.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_ICR-01_1_neighbor_6_size_32.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-01_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-01_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-01_3_neighbor_6_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-02_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-02_2_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-02_3_neighbor_6_size_32.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-03_1_neighbor_6_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-03_2_neighbor_6_size_32.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\ae\autoencoder_e100_SMT-04_1_neighbor_6_size_32.txt']
    }


    # 26-neighborhood
    # predicts = {
    #     'EMC-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-01_1'],
    #     'EMC-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
    #     'EMC-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-03_1'],
    #     'ICR-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_ICR-01_1'],
    #     'SMT-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_3'],
    #     'SMT-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_1',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_2',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_3'],
    #     'SMT-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_1_neighbor26.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_2_neighbor26.txt'],
    #     'SMT-04': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    # }
    return predicts


def get_ncc_mb_predicts():

    n6_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-01_1'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-03_1'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_ICR-01_1'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_3'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_3'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_1_neighbor26.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_2_neighbor26.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    }

    n26_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-01_1'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_EMC-03_1'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_ICR-01_1'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-01_3'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-02_3'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_1_neighbor26.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\ae\ResNetAutoEncoder3D_SMT-03_2_neighbor26.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    }


def get_multi_branch_n26_predicts():
    # predicts = {
    #     'EMC-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-01_1_reg32_dis64.txt'],
    #     'EMC-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_1_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_2_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_3_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_4_spatial_0.001_size_32.txt'],
    #     'EMC-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-03_1_spatial_0.001_size_32.txt'],
    #     'ICR-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_ICR-01_1_spatial_0.001_size_32.txt'],
    #     'SMT-01': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_1_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32.txt'],
    #     'SMT-02': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_1_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_2_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_3_spatial_0.001_size_32.txt'],
    #     'SMT-03': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-03_1_spatial_0.001_size_32.txt',
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-03_2_spatial_0.001_size_32.txt'],
    #     'SMT-04': [
    #         r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.001_size_32.txt']
    # }

    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-01_1_reg32_dis64_1-9.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_3_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_4_reg32_dis64_2-8.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-03_1_reg32_dis64_2-8.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_ICR-01_1_reg32_dis64_2-8.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_1_reg32_dis64_6-4.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_3_reg32_dis64_2-8.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_2_reg32_dis64_2-8.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.001_size_32.txt']
    }
    return predicts

if __name__ == '__main__':
    series = ['EMC-01', 'EMC-02', 'EMC-03', 'ICR-01', 'SMT-01', 'SMT-02', 'SMT-03', 'SMT-04']
    predicts = {
        'EMC-01': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e290_EMC-01_1_spatial_0.001_size_32.txt'],
        'EMC-02': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_1_spatial_0.0.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_2_spatial_0.0.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-02_3_size_32.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_EMC-02_4_size_32.txt'],
        'EMC-03': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e210_EMC-03_1_spatial_0.0_size_24.txt'],
        'ICR-01': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e290_ICR-01_1_spatial_0.001_size_32.txt'],
        'SMT-01': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-01_1_spatial_0.0.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.0_size_24.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e290_SMT-01_3_spatial_0.001_size_32.txt'],
        'SMT-02': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e290_SMT-02_1_spatial_0.001_size_32.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_SMT-02_2_size_32.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_SMT-02_3_size_32.txt'],
        'SMT-03': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_SMT-03_1_size_32.txt',
                   r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\multi_branch\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e155_SMT-03_2_size_32.txt'],
        'SMT-04': [r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.0_size64.txt']
    }
    dcn_9c_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-01_1_dcn_AutoEncoderV1_9c_new_e300'
            #r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-01_1_dcn_AutoEncoderV1_9c_105'
        ],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-02_1_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-02_2_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-02_3_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-02_4_dcn_AutoEncoderV1_9c_105'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_EMC-03_1_dcn_AutoEncoderV1_9c_105'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_ICR-01_1_dcn_AutoEncoderV1_9c_105'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-01_1_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-01_2_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-01_3_dcn_AutoEncoderV1_9c_105'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-02_1_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-02_2_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-02_3_dcn_AutoEncoderV1_9c_105'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-03_1_dcn_AutoEncoderV1_9c_105',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-03_2_dcn_AutoEncoderV1_9c_105'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\auto-encoder\dcn\AutoEncoderV1_9c_z_SMT-04_1_dcn_AutoEncoderV1_9c_105']
    }
    moco_predicts = get_moco_predicts()
    mb_b6_predicts = get_multi_branch_n6_predicts()
    mb_b6_ncc_predicts = get_multi_branch_n6_ncc_predicts()
    ae_b6_predicts = get_ae_n6_predicts()

    ae_predicts = get_ae_predicts()

    best_predicts = get_multi_branch_n26_predicts()

    labels = get_gt_file(dataset_path)
    #calculate_total_error(mb_b6_predicts, labels)

    #calculate_total_error(moco_predicts, labels)

    #calculate_total_error(ae_b6_predicts, labels)
    print("---------------------------ae------------------------")
    calculate_total_error(ae_predicts, labels)
    print("---------------------------ae 6-neighborhood------------------------")
    calculate_total_error(ae_b6_predicts, labels)
    # print("---------------------------multi branch------------------------")
    # calculate_total_error(mb_b6_predicts, labels)
    # print("---------------------------multi branch ncc------------------------")
    # calculate_total_error(mb_b6_ncc_predicts, labels)
    print("---------------------------best predicts------------------------")
    calculate_total_error(best_predicts, labels)
    print("---------------------------dcn 9c--------------------------------")
    calculate_total_error(dcn_9c_predicts, labels)

    # n6_ae_predicts_folder = r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\6邻域\单autoencoder分支"
    # n26_ae_predicts_folder = r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\26邻域\单autoencoder分支"
    # for key, value in ae_predicts.items():
    #     print("Start copying {}".format(key))
    #     copy_specific_files(value, n26_ae_predicts_folder, key)
    #
    # for key, value in ae_b6_predicts.items():
    #     print("Start copying {}".format(key))
    #     copy_specific_files(value, n6_ae_predicts_folder, key)
