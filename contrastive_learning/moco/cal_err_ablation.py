import numpy as np
import math
import os
from glob import glob

dataset_path = r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D"

target_nums = {
    'EMC-01': 1,
    'EMC-02': 4,
    'EMC-03': 1,
    'ICR-01': 1,
    'SMT-01': 3,
    'SMT-02': 3,
    'SMT-03': 2,
    'SMT-04': 1
}

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


def get_all_predicts(file_path):
    all_predicts = sorted(glob(os.path.join(file_path, "*.txt")))
    seq_predicts = dict()
    start = 0
    for seq, num in target_nums.items():
        end = start + num
        seq_predicts[seq] = []
        seq_predicts[seq].extend(all_predicts[start:end])
        start = end

    return seq_predicts



def calculate_total_error(predicts, labels):
    series = labels.keys()
    errors = []
    for s in series:
        # if s != 'SMT-03':
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
    ae_predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-01_1'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_3',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-02_4'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_EMC-03_1'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_ICR-01_1'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-01_3'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_2',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-02_3'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_1',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-03_2'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\auto-encoder\3d\ResNetAutoEncoder3D_SMT-04_1']
    }
    return ae_predicts

def get_moco_ncc_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-01_1_spatial_0.001_size_32_ncc.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-02_1_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-02_2_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-02_3_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-02_4_spatial_0.001_size_32_ncc.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_EMC-03_1_spatial_0.001_size_32_ncc.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_ICR-01_1_spatial_0.001_size_32_ncc.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-01_1_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32_ncc.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-02_1_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-02_2_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-02_3_spatial_0.001_size_32_ncc.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-03_1_spatial_0.001_size_32_ncc.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-03_2_spatial_0.001_size_32_ncc.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\moco\size64\moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.001_size_32_ncc.txt']
    }
    return predicts


def get_moco_ncc_n26_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-01_1_spatial_0.001_size_32.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_3_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-02_4_spatial_0.001_size_32.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_EMC-03_1_spatial_0.001_size_32.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_ICR-01_1_spatial_0.001_size_32.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_3_spatial_0.001_size_32.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-03_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-03_2_spatial_0.001_size_32.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.001_size_32.txt']
    }
    return predicts

def get_moco_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-01_1_spatial_0.0.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_1_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_2_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_3_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_EMC-02_4_spatial_0.0.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e210_EMC-03_1_spatial_0.0_size_24.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_ICR-01_1_spatial_0.0.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-01_1_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.0_size_24.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-01_3_spatial_0.0.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_1_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_2_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-02_3_spatial_0.0.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-03_1_spatial_0.0.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-03_2_spatial_0.0.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\moco\moco_encoder_q_resnet50_e155_SMT-04_1_spatial_0.0.txt']
    }
    return predicts


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
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-01_1_reg32_dis64_1-9.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_3_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_4_reg32_dis64_2-8.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-03_1_reg32_dis64_2-8.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_ICR-01_1_reg32_dis64_2-8.txt'],
        'SMT-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_1_reg32_dis64_6-4.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_2_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32.txt'],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-02_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_3_reg32_dis64_2-8.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\multi_branch_test\ncc\ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_2_reg32_dis64_2-8.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-04_1_spatial_0.001_size_32.txt']
    }
    return predicts

def get_multi_branch_n6_predicts():
    predicts = {
        'EMC-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-01_1_reg32_dis64_2-8.txt'],
        'EMC-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_3_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-02_4_reg32_dis64_2-8.txt'],
        'EMC-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_EMC-03_1_reg32_dis64_2-8.txt'],
        'ICR-01': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_ICR-01_1_reg32_dis64_2-8.txt'],
        'SMT-01': [
            # r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_1_reg32_dis64_2-8.txt',
            # r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_2_reg32_dis64_2-8.txt',
            # r'D:\pycharmProject\pythonProject1\coding\3dTracking\track_test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_3_reg32_dis64_2-8.txt'
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_1_spatial_0.001_size_32.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-01_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_26\moco\size64\n26_moco_encoder_q_resnet50_e210_SMT-01_3_spatial_0.001_size_32.txt'
        ],
        'SMT-02': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_2_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-02_3_reg32_dis64_2-8.txt'],
        'SMT-03': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_1_reg32_dis64_2-8.txt',
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-03_2_reg32_dis64_2-8.txt'],
        'SMT-04': [
            r'D:\pycharmProject\pythonProject1\coding\3dTracking\test\result\ablation\neighbor_6\multi_branch_test\ncc\n6ResNetAutoEncoder3D_3d_100-moco_encoder_q_resnet50_e210_SMT-04_1_reg32_dis64_2-8.txt']
    }

    return predicts

if __name__ == '__main__':
    series = ['EMC-01', 'EMC-02', 'EMC-03', 'ICR-01', 'SMT-01', 'SMT-02', 'SMT-03', 'SMT-04']


    # moco_predicts = get_moco_predicts()
    # moco_ncc_predicts = get_moco_ncc_predicts()
    # moco_ncc_n26_predicts = get_moco_ncc_n26_predicts()
    # multi_branch_n26_predicts = get_multi_branch_n26_predicts()
    #
    # multi_branch_n6_predicts = get_multi_branch_n6_predicts()
    #
    # labels = get_gt_file(dataset_path)
    # print("------------moco-----------")
    # calculate_total_error(moco_predicts, labels)
    # print("------------moco ncc n6-----------")
    # calculate_total_error(moco_ncc_predicts, labels)
    # print("------------moco ncc n26-----------")
    # calculate_total_error(moco_ncc_n26_predicts, labels)
    #
    # print("------------multi branch ncc size64 n26-----------")
    # calculate_total_error(multi_branch_n26_predicts, labels)
    #
    # print("------------multi branch ncc size64 n6-----------")
    # calculate_total_error(multi_branch_n6_predicts, labels)

    labels = get_gt_file(dataset_path)

    n6_ae_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\6邻域\单autoencoder分支")
    print("------------n6_ae_predicts-----------")
    calculate_total_error(n6_ae_predicts, labels)

    n26_ae_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\26邻域\单autoencoder分支")
    print("------------n6_ae_n26_predicts-----------")
    calculate_total_error(n26_ae_predicts, labels)

    n6_moco_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\6邻域\单moco分支")
    print("------------n6_moco_predicts-----------")
    calculate_total_error(n6_moco_predicts, labels)

    n26_moco_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\26邻域\单moco分支")
    print("------------n26_moco_predicts-----------")
    calculate_total_error(n26_moco_predicts, labels)

    n6_mb_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\6邻域\双分支")
    print("------------n6_mb_predicts-----------")
    calculate_total_error(n6_mb_predicts, labels)

    n26_mb_predicts = get_all_predicts(r"C:\Users\hello\Desktop\0_毕业论文\3d\消融\26邻域\双分支")
    print("------------n26_mb_predicts-----------")
    calculate_total_error(n26_mb_predicts, labels)



