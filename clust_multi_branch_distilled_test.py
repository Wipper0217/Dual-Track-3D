"""
Multi-branch tracking test script — distilled version
The tracking logic is the same as clust_multi_branch_test.py (autoencoder branch + MoCo branch, weighted similarity)

Difference:
- The MoCo branch uses a knowledge-distilled ResNet-18 student model
  instead of the original MoCo ResNet-50 teacher model
- The autoencoder branch stays unchanged (ResNetAutoEncoder3D, cross-entropy/cosine similarity)

Branch description:
- Autoencoder branch: ResNetAutoEncoder3D → cosine similarity (LAMBDA_REG=0.2)
- MoCo branch: distilled ResNet-18 StudentMoCoEncoder → NCC (LAMBDA_DIS=0.8)

Usage:
    python clust_multi_branch_distilled_test.py \
        --model_autoencoder trained_models/core-track/ResNetAutoEncoder3D_3d_100.pth \
        --model_distilled snapshot/moco_distillation_v3/student_encoder_v3_best.pth \
        --clust_path /root/data-pub/ultrasound/3DUS/CLUST/clust_trainset_3d
"""
import sys
import os
import argparse
import logging
import time
from glob import glob

import torch
import numpy as np
import SimpleITK as sitk
import random

from core.config_multi_branch import cfg
from utils.model_load import load_pretrain
from model.model_3d.resnet.resnet_ae_3d import ResNetAutoEncoder3D
from moco_test_new.model.student_encoder import StudentMoCoEncoder
from moco_test_new.model.student_encoder_32 import StudentMoCoEncoder32
from tracker.MultiBranchTracker_18 import MultiBranchTracker
from utils.metrics import error_metrics, format_metrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('multi_branch_distilled_test')

parser = argparse.ArgumentParser(description='Multi-branch tracking test with distilled MoCo student')
parser.add_argument('--cfg', type=str,
                    default='experiments/moco/config_core_track.yaml',
                    help='configuration of tracking')
parser.add_argument('--model_autoencoder', type=str,
                    default='trained_models/core-track/ResNetAutoEncoder3D_3d_100.pth',
                    help='autoencoder branch model path')
parser.add_argument('--model_distilled', type=str,
                    default='snapshot/moco_distillation_v3/student_encoder_v3_best.pth',
                    help='distilled student encoder (ResNet-18) model path')
parser.add_argument('--clust_path', type=str,
                    default='/root/data-pub/ultrasound/3DUS/CLUST/clust_trainset_3d',
                    help='CLUST dataset path')
parser.add_argument('--seed', type=int, default=3407,
                    help='random seed')
parser.add_argument('--use_optimized', action='store_true',
                    help='use OptimizedMultiBranchTracker with batch GPU inference')
parser.add_argument('--use_amp', action='store_true',
                    help='use mixed precision (float16) for forward pass')
parser.add_argument('--use_compile', action='store_true',
                    help='use torch.compile for model acceleration')
parser.add_argument('--use_adaptive_fusion', action='store_true',
                    help='use adaptive fusion weights based on branch discriminability')
parser.add_argument('--top_k_fusion', type=int, default=0,
                    help='if >0, use top-K fusion: MoCo re-ranks only top-K AE candidates')
parser.add_argument('--top_k_weight_ae', type=float, default=0.5,
                    help='AE weight in top-K fusion (0-1)')
parser.add_argument('--lambda_reg', type=float, default=None,
                    help='override AE branch weight (default: from config)')
parser.add_argument('--lambda_dis', type=float, default=None,
                    help='override MoCo branch weight (default: from config)')
parser.add_argument('--use_weighted_ncc', action='store_true',
                    help='use variance-based channel weighting for per-channel NCC')
parser.add_argument('--dis_img_size', type=int, default=64,
                    help='MoCo branch input size (32 or 64)')
args = parser.parse_args()


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_student_encoder(model, pretrained_path, device):
    """
    Load StudentMoCoEncoder weights from a MoCoStudentScorer checkpoint.

    The key format in the checkpoint is:
      encoder.conv1.weight, encoder.bn1.weight, ...
      feature_projection.0.weight, ...

    The key format of StudentMoCoEncoder is:
      conv1.weight, bn1.weight, ...

    Therefore the 'encoder.' prefix must be stripped and feature_projection-related keys filtered out.
    """
    logger.info(f"Loading distilled student encoder from {pretrained_path}")
    assert os.path.exists(pretrained_path), f"Model file not found: {pretrained_path}"

    pretrained_dict = torch.load(pretrained_path, map_location=device)

    # handle possible key prefixes
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    model_keys = set(model.state_dict().keys())

    # strip the 'encoder.' prefix and filter out non-encoder weights
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('encoder.'):
            new_key = k[len('encoder.'):]
            if new_key in model_keys:
                filtered_dict[new_key] = v

    matched = len(filtered_dict)
    missing = len(model_keys - set(filtered_dict.keys()))
    logger.info(f"  Matched keys: {matched}, Missing keys: {missing}")

    if missing > 0:
        logger.warning(f"  Missing keys: {list(model_keys - set(filtered_dict.keys()))[:5]}")

    model.load_state_dict(filtered_dict, strict=False)

    # verify weight loading
    first_param = list(model.parameters())[0]
    logger.info(f"  First param norm: {first_param.data.norm().item():.6f}")

    return model


def clust_track(path, tracker, model_name, series_index=-1, target_index=-1):
    series_list = sorted([s for s in os.listdir(path)
                          if os.path.isdir(os.path.join(path, s))])
    all_errors = []
    total_elapsed = 0.0
    total_track_frames = 0
    for idx, series in enumerate(series_list):
        if series_index != -1 and idx != series_index:
            continue
        series_path = os.path.join(path, series)
        series_error, elapsed, track_frames = series_track(series_path, tracker, model_name, target_index)
        all_errors = series_error + all_errors
        total_elapsed += elapsed
        total_track_frames += track_frames
    all_errors = np.asarray(all_errors, dtype=float)
    m = error_metrics(all_errors, drop_first=False)
    te95th = m["p95"]
    avg_fps = total_track_frames / total_elapsed if total_elapsed > 0 else 0
    avg_ms = total_elapsed / total_track_frames * 1000 if total_track_frames > 0 else 0
    print("=" * 60)
    print("Total mean(mm): {}, std(mm): {}, te95th(mm): {}, max(mm): {}".format(
        m["mean"], m["std"], te95th, m["max"]))
    print("Total tracked frames: {}, total time: {:.2f}s".format(total_track_frames, total_elapsed))
    print("Average tracking speed: {:.2f} fps ({:.2f} ms/frame)".format(avg_fps, avg_ms))
    print("=" * 60)


def series_track(series_path, tracker, model_name, index=-1):
    series_name = os.path.basename(series_path)
    data_path = os.path.join(series_path, 'Data')
    anno_path = os.path.join(series_path, 'Annotation')
    data_files = sorted(glob(os.path.join(data_path, '*.mha')))
    anno_files = sorted(glob(os.path.join(anno_path, '*.txt')))
    res_file = glob(os.path.join(series_path, 'res.txt'))

    # resolution settings
    if series_name.startswith('EMC'):
        res = np.array([1.14, 0.59, 1.19])
    elif series_name.startswith('ICR'):
        res = np.array([0.31, 0.51, 0.67])
    elif series_name.startswith('SMT'):
        res = np.array([0.7, 0.7, 0.7])
    else:
        res = np.loadtxt(res_file[0])

    target_num = len(anno_files)
    if index != -1:
        targets = [index]
    else:
        targets = range(target_num)
    print("Start tracking {}".format(series_name))
    total_errors = []
    total_elapsed = 0.0
    total_track_frames = 0
    for i in targets:
        anno_file = anno_files[i]
        print("Tracking target {} in {}".format(i + 1, series_name))
        anno = np.loadtxt(anno_file)
        anno_frames = anno[:, 0].tolist()
        center = anno[0, 1:].tolist()

        predict_positions, elapsed, num_track_frames = model_track(data_files, tracker, center)
        total_elapsed += elapsed
        total_track_frames += num_track_frames

        errors = []
        for idx in range(len(anno_frames)):
            frame_idx = int(anno_frames[idx]) - 1
            predict = predict_positions[frame_idx]
            label = anno[idx, 1:]
            error = predict - label
            error_mm = error * res
            error = np.sqrt(error_mm[0] ** 2 + error_mm[1] ** 2 + error_mm[2] ** 2)
            errors.append(error)
            total_errors.append(error)
        m = error_metrics(errors, drop_first=False)
        print("Target {} mean(mm): {}, std(mm): {}, te95th(mm): {}, max(mm): {}".format(
            i + 1, m["mean"], m["std"], m["p95"], m["max"]))
    tm = error_metrics(total_errors, drop_first=False)
    avg_fps = total_track_frames / total_elapsed if total_elapsed > 0 else 0
    avg_ms = total_elapsed / total_track_frames * 1000 if total_track_frames > 0 else 0
    print("Series {} mean(mm): {}, std(mm): {}, te95th(mm): {}, max(mm): {}, avg_speed: {:.2f} fps ({:.2f} ms/frame)".format(
        series_name, tm["mean"], tm["std"], tm["p95"], tm["max"], avg_fps, avg_ms))
    return total_errors, total_elapsed, total_track_frames


def model_track(data_files, tracker, center):
    predict_positions = []
    start = time.time()
    for idx, file in enumerate(data_files):
        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)
        if idx == 0:
            predict_positions.append(center)
            tracker.new_init(image, center)
            start = time.time()
        else:
            predict, similarity = tracker.new_track(image)
            center = predict
            predict_positions.append(center)
    end = time.time()
    elapsed = end - start
    num_track_frames = len(data_files) - 1
    fps = num_track_frames / elapsed
    ms_per_frame = elapsed / num_track_frames * 1000
    print("track speed: {:.2f} fps ({:.2f} ms/frame)".format(fps, ms_per_frame))
    return predict_positions, elapsed, num_track_frames



if __name__ == '__main__':
    set_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # load config
    cfg.merge_from_file(args.cfg)
    img_size = cfg.DATASET.IMG_SIZE
    reg_img_size = cfg.TRACKER.REG_IMG_SIZE
    dis_img_size = cfg.TRACKER.DIS_IMG_SIZE
    lambda_reg = cfg.TRACKER.LAMBDA_REG
    lambda_dis = cfg.TRACKER.LAMBDA_DIS

    # CLI overrides for lambda values
    if args.lambda_reg is not None:
        lambda_reg = args.lambda_reg
    if args.lambda_dis is not None:
        lambda_dis = args.lambda_dis

    # CLI override for dis_img_size
    if args.dis_img_size != 64:
        dis_img_size = args.dis_img_size

    # ========== 1. Load the autoencoder branch (unchanged) ==========
    autoencoder_model = ResNetAutoEncoder3D(num_layers=18)
    autoencoder_model = load_pretrain(autoencoder_model, args.model_autoencoder).to(device)
    logger.info(f"AutoEncoder loaded: {args.model_autoencoder}")

    # ========== 2. Load the distilled MoCo branch (ResNet-18 instead of ResNet-50) ==========
    if dis_img_size == 32:
        distilled_encoder = StudentMoCoEncoder32(in_channels=1)
        logger.info(f"Using StudentMoCoEncoder32 (32³ input)")
    else:
        distilled_encoder = StudentMoCoEncoder(in_channels=1)
        logger.info(f"Using StudentMoCoEncoder (64³ input)")
    distilled_encoder = load_student_encoder(distilled_encoder, args.model_distilled, device)
    distilled_encoder = distilled_encoder.to(device)
    logger.info(f"Distilled StudentMoCoEncoder loaded: {args.model_distilled}")

    # ========== 3. Build the multi-branch tracker ==========
    if args.use_optimized:
        from tracker.OptimizedMultiBranchTracker import OptimizedMultiBranchTracker
        tracker = OptimizedMultiBranchTracker(
            model_autoencoder=autoencoder_model,
            model_encoder=distilled_encoder,
            device=device,
            reg_img_size=reg_img_size,
            dis_img_size=dis_img_size,
            lambda_reg=lambda_reg,
            lambda_dis=lambda_dis,
            use_amp=args.use_amp,
            use_compile=args.use_compile,
            use_adaptive_fusion=args.use_adaptive_fusion,
            top_k_fusion=args.top_k_fusion,
            top_k_weight_ae=args.top_k_weight_ae,
            use_weighted_ncc=args.use_weighted_ncc,
        )
    else:
        tracker = MultiBranchTracker(
            model_autoencoder=autoencoder_model,
            model_encoder=distilled_encoder,
            img_size=img_size,
            device=device,
            reg_img_size=reg_img_size,
            dis_img_size=dis_img_size,
            lambda_reg=lambda_reg,
            lambda_dis=lambda_dis
        )

    model_name = "DistilledStudent_R18"
    logger.info(
        f"Tracker: {tracker.__class__.__name__}, "
        f"img_size={img_size}, reg_img_size={reg_img_size}, dis_img_size={dis_img_size}, "
        f"lambda_reg={lambda_reg}, lambda_dis={lambda_dis}"
    )

    # ========== 4. Evaluate on the CLUST dataset ==========
    clust_track(args.clust_path, tracker, model_name)
