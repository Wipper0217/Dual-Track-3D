import argparse
import logging
import time
from glob import glob

import torch
import numpy as np
import os
import SimpleITK as sitk
import random

from monai.transforms import Compose, ScaleIntensityd

from core.config_multi_branch import cfg
from contrastive_learning.encoder.resnet.resnet import resnet50
from utils.model_load import load_pretrain
from tracker.MultiBranchTracker_18 import MultiBranchTracker
from model.model_3d.resnet.resnet_ae_3d import ResNetAutoEncoder3D



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('clust 3d track_test')

predict_result_save_path = r"/track_test\result\ablation\neighbor_26\multi_branch_test\ncc"
parser = argparse.ArgumentParser(description='autoencoder tracking')
parser.add_argument('--cfg', type=str, default=r'E:\tracker_3d\core-track\experiments\moco\config_core_track.yaml',
                    help='configuration of tracking')
parser.add_argument('--model_autoencoder', type=str,
                    #default=r"D:\pycharmProject\pythonProject1\coding\3dTracking\train_models\auto-encoder\testset\dataset_new\3d\resnet18\ResNetAutoEncoder3D_3d_100.pth",
                    default=r"E:\tracker_3d\core-track\trained_models\core-track\ResNetAutoEncoder3D_3d_100.pth",
                    help='autoencoder branch model name')
parser.add_argument('--model_encoder', type=str,
                    default=r"E:\tracker_3d\core-track\trained_models\core-track\moco_encoder_q_resnet50_e210.pth",
                    help='encoder branch model name')
parser.add_argument('--seed', type=int, default=123568,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()
def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def pre_process(image):
    data_dict = {
        "image": image
    }
    trasnform = Compose([
            # Normalize to [0, 1]
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])
    transform_img = trasnform(data_dict)
    return transform_img['image'].as_tensor()

def clust_track(path, tracker, model_name, series_index=-1, target_index=-1):

    series_list = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
    all_errors = []
    for idx, series in enumerate(series_list):
        if series_index != -1 and idx != series_index:
            continue
        series_path = os.path.join(path, series)
        series_error = series_track(series_path, tracker, model_name, target_index)
        all_errors = series_error + all_errors
    error_mean = np.mean(all_errors)
    error_std = np.std(all_errors)
    te95th = np.percentile(all_errors, 95)
    print("Total mean(mm): {}, std(mm): {}, te95th(mm): {}".format(error_mean, error_std, te95th))

def series_track(series_path, tracker, model_name, index=-1):
    series_name = os.path.basename(series_path)
    # if series_name not in ['EMC-01']:
    #     return []
    data_path = os.path.join(series_path, 'Data')
    anno_path = os.path.join(series_path, 'Annotation')
    data_files = sorted(glob(os.path.join(data_path, '*.mha')))
    anno_files = sorted(glob(os.path.join(anno_path, '*.txt')))
    res_file = glob(os.path.join(series_path, 'res.txt'))

    size_files = glob(os.path.join(series_path, 'size.txt'))
    res = np.loadtxt(res_file[0])
    target_num = len(anno_files)
    if index != -1:
        targets = [index]
    else:
        targets = range(target_num)
    print("Start tracking {}".format(series_name))
    total_errors = []
    for i in targets:
        anno_file = anno_files[i]
        print("Tracking target {} in {}".format(i+1, series_name))
        anno = np.loadtxt(anno_file)
        anno_frames = anno[:, 0].tolist()
        # if i > 0:
        #     break
        center = anno[0, 1:].tolist()
        #print(center)
        predict_positions = model_track(data_files, tracker, center)
        reg_param, dis_param = get_params(cfg.TRACKER.LAMBDA_REG, cfg.TRACKER.LAMBDA_DIS)
        file_name = "n6{}_{}_{}_reg{}_dis{}_{}-{}.txt".format(model_name, series_name, i+1,
                                                            cfg.TRACKER.REG_IMG_SIZE, cfg.TRACKER.DIS_IMG_SIZE,
                                                            reg_param, dis_param)
        #np.savetxt(os.path.join(predict_result_save_path, file_name), predict_positions)
        errors = []
        #print(len(predict_positions))
        for idx in range(len(anno_frames)):
            frame_idx = int(anno_frames[idx]) - 1
            predict = predict_positions[frame_idx]
            label = anno[idx, 1:]
            error = predict - label
            error_mm = error * res
            error = np.sqrt(error_mm[0]**2 + error_mm[1]**2 + error_mm[2]**2)
            errors.append(error)
            total_errors.append(error)
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        te95th = np.percentile(errors, 95)
        print("Target {} mean(mm): {}, std(mm): {}, te95th(mm): {}".format(i+1, error_mean, error_std, te95th))
    total_error_mean = np.mean(total_errors)
    total_error_std = np.std(total_errors)
    total_te95th = np.percentile(total_errors, 95)
    print("Series {} mean(mm): {}, std(mm): {}, te95th(mm): {}".format(series_name, total_error_mean, total_error_std, total_te95th))
    return total_errors

def model_track(data_files, tracker, center, track_visualized=False):
    predict_positions = []
    #print("init position={}".format(center))
    start = time.time()
    for idx, file in enumerate(data_files):

        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)
        if idx == 0:
            state = center
            predict_positions.append(center)
            #tracker.init(image, center)
            tracker.new_init(image, center)
            start = time.time()
        else:

            #predict = tracker.my_track(image, search_radius=3, next_search_amount=5)
            #predict, similarity = tracker.track(image)
            predict, similarity = tracker.new_track(image)
            center = predict
            #print("No.{} frame track position={}, distance={}".format(idx + 1, predict, distance))
            if track_visualized:
                print("No.{} frame predict position = {}".format(idx+1, center))

            predict_positions.append(center)
    end = time.time()
    print("track speed: {} fps".format((len(data_files)-1)/(end-start)))
    return predict_positions

def get_params(lambda_reg, lambda_dis):
    reg = str(lambda_reg).split('.')
    dis = str(lambda_dis).split('.')
    reg_1 = int(reg[0])
    reg_2 = int(reg[1])
    dis_1 = int(dis[0])
    dis_2 = int(dis[1])

    if reg_1 == 1:
        reg_param = reg_1
    else:
        reg_param = reg_2

    if dis_1 == 1:
        dis_param = dis_1
    else:
        dis_param = dis_2

    return reg_param, dis_param


if __name__ == '__main__':
    """
    series = ['EMC-01', 'EMC-02', 'EMC-03', 'ICR-01', 'SMT-01', 'SMT-02', 'SMT-03', 'SMT-04']
    """
    set_seed(3407)
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    cfg.merge_from_file(args.cfg)
    img_size = cfg.DATASET.IMG_SIZE
    method = cfg.DATASET.METHOD
    normalize = cfg.DATASET.NORMALIZE
    lambda_spatial = cfg.TRACKER.LAMBDA_SPATIAL
    mlp = True
    reg_img_size = cfg.TRACKER.REG_IMG_SIZE
    dis_img_size = cfg.TRACKER.DIS_IMG_SIZE

    lambda_reg = cfg.TRACKER.LAMBDA_REG
    lambda_dis = cfg.TRACKER.LAMBDA_DIS

    clust_path = r"E:\1DATA\CLUST\CLUST_3D\clust_trainset_3d"

    autoencoder_model = ResNetAutoEncoder3D(num_layers=18)
    encoder_model = resnet50()
    model_autoencoder_path = args.model_autoencoder
    model_encoder_path = args.model_encoder
    autoencoder_model = load_pretrain(autoencoder_model, model_autoencoder_path).to(device)
    encoder_model = load_pretrain(encoder_model, model_encoder_path).to(device)
    tracker = MultiBranchTracker(
        model_autoencoder=autoencoder_model,
        model_encoder=encoder_model,
        img_size=img_size,
        device=device,
        reg_img_size=reg_img_size,
        dis_img_size=dis_img_size,
        lambda_reg=lambda_reg,
        lambda_dis=lambda_dis
    )
    select_tracker = tracker
    # logger.info("load model from {}".format(args.model))
    model_1 = os.path.basename(args.model_autoencoder).split('.')[0]
    model_2 = os.path.basename(args.model_encoder).split('.')[0]
    model_name = "{}-{}".format(model_1, model_2)
    logger.info("tracker: {}, lambda_spatial={}, img_size={}, reg_img_size={}, dis_img_size={}, lambda_reg={}, lambda_dis={}".format(select_tracker.__class__.__name__,
                                                                     lambda_spatial,
                                                                     img_size, reg_img_size, dis_img_size, lambda_reg, lambda_dis))
    series_index = -1
    target_index = -1
    clust_track(clust_path, select_tracker, model_name, series_index, target_index)
    #res_file = glob(os.path.join(r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D\SMT-01", 'res.txt'))
    #print(res_file)



