"""
MoCo student network tracking test script V3 (32³ version)
Uses only the student ResNet-18 encoder, input 32³, uses NCC as the similarity metric

Usage:
    python moco_test_new/moco_student_track_test_32.py \
        --student_model snapshot/moco_distillation_v3/student_encoder_v3_best.pth
"""
import sys
import os

# Ensure the project root directory is on the Python module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import time
from glob import glob

import torch
import numpy as np
import SimpleITK as sitk
import random

from moco_test_new.model.student_encoder_32 import MoCoStudentScorer32
from moco_test_new.tracker.student_moco_tracker_32 import StudentMoCoTracker32

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('moco_student_track_test_v3_32')

parser = argparse.ArgumentParser(description='MoCo Student Encoder Tracking Test V3 (32³)')
parser.add_argument('--student_model', type=str,
                    default="snapshot/moco_distillation_v3/student_encoder_v3_best.pth",
                    help='distilled student encoder model path')
parser.add_argument('--clust_path', type=str,
                    default="/root/data-pub/ultrasound/3DUS/CLUST/clust_trainset_3d",
                    help='CLUST dataset path')
parser.add_argument('--seed', type=int, default=3407,
                    help='random seed')
parser.add_argument('--img_size', type=int, default=32,
                    help='input patch size (default: 32)')
parser.add_argument('--step', type=int, default=1,
                    help='search step size')
parser.add_argument('--neighbor_type', type=str, default='18',
                    choices=['6', '18', '26'],
                    help='neighbor type for greedy search')
parser.add_argument('--verbose', action='store_true',
                    help='show detailed tracking process for the first series')
args = parser.parse_args()


def set_seed(seed=3407):
    """Set the random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clust_track(path, tracker, model_name, series_index=-1, target_index=-1,
                verbose=False):
    """Perform tracking evaluation on the CLUST dataset"""
    series_list = sorted([s for s in os.listdir(path)
                          if os.path.isdir(os.path.join(path, s))])
    all_errors = []
    all_fps = []

    for idx, series in enumerate(series_list):
        if series_index != -1 and idx != series_index:
            continue
        series_path = os.path.join(path, series)
        is_verbose = verbose and idx == 0
        if is_verbose:
            print("\n" + "#"*60)
            print("# VERBOSE MODE: Showing detailed tracking for first series")
            print("#"*60)
        series_error, fps = series_track(series_path, tracker, model_name, target_index,
                                         verbose=is_verbose)
        all_errors = series_error + all_errors
        all_fps.append(fps)

    error_mean = np.mean(all_errors)
    error_std = np.std(all_errors)
    te95th = np.percentile(all_errors, 95)
    avg_fps = np.mean(all_fps) if all_fps else 0
    print("Total mean(mm): {}, std(mm): {}, te95th(mm): {}".format(error_mean, error_std, te95th))
    print("Average FPS: {}".format(avg_fps))


def series_track(series_path, tracker, model_name, index=-1, verbose=False):
    """Track a single series"""
    series_name = os.path.basename(series_path)
    data_path = os.path.join(series_path, 'Data')
    anno_path = os.path.join(series_path, 'Annotation')
    data_files = sorted(glob(os.path.join(data_path, '*.mha')))
    anno_files = sorted(glob(os.path.join(anno_path, '*.txt')))

    # Resolution settings
    if series_name.startswith('EMC'):
        res = np.array([1.14, 0.59, 1.19])
    elif series_name.startswith('ICR'):
        res = np.array([0.31, 0.51, 0.67])
    elif series_name.startswith('SMT'):
        res = np.array([0.7, 0.7, 0.7])
    else:
        res_file = glob(os.path.join(series_path, 'res.txt'))
        res = np.loadtxt(res_file[0])

    target_num = len(anno_files)
    if index != -1:
        targets = [index]
    else:
        targets = range(target_num)

    print("Start tracking {}".format(series_name))
    total_errors = []
    total_fps = 0
    for i in targets:
        anno_file = anno_files[i]
        print("Tracking target {} in {}".format(i + 1, series_name))
        anno = np.loadtxt(anno_file)
        anno_frames = anno[:, 0].tolist()
        center = anno[0, 1:].tolist()

        is_verbose = verbose and i == 0
        if is_verbose:
            print("=" * 50)
            print("DETAILED TRACKING LOG (first target only)")
            print("Initial position: ({:.1f}, {:.1f}, {:.1f})".format(center[0], center[1], center[2]))
            print("Total annotated frames: {}".format(len(anno_frames)))
            print("=" * 50)

        predict_positions, fps = model_track(data_files, tracker, center, verbose=is_verbose)
        total_fps = fps

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

        if is_verbose:
            print("=" * 50)
            print("TRACKING RESULT SUMMARY (first target)")
            print("=" * 50)
            for idx in range(len(anno_frames)):
                frame_idx = int(anno_frames[idx]) - 1
                predict = predict_positions[frame_idx]
                label = anno[idx, 1:]
                print("    Frame {:3d}: pred=({:7.2f}, {:7.2f}, {:7.2f})  gt=({:7.2f}, {:7.2f}, {:7.2f})  err={:.2f}mm".format(
                    frame_idx+1, predict[0], predict[1], predict[2],
                    label[0], label[1], label[2], errors[idx]))

        error_mean = np.mean(errors)
        error_std = np.std(errors)
        te95th = np.percentile(errors, 95)
        print("Target {} mean(mm): {}, std(mm): {}, te95th(mm): {}".format(
            i + 1, error_mean, error_std, te95th))

    total_error_mean = np.mean(total_errors)
    total_error_std = np.std(total_errors)
    total_te95th = np.percentile(total_errors, 95)
    print("Series {} mean(mm): {}, std(mm): {}, te95th(mm): {}".format(
        series_name, total_error_mean, total_error_std, total_te95th))
    return total_errors, total_fps


def model_track(data_files, tracker, center, verbose=False):
    """Frame-by-frame tracking"""
    predict_positions = []
    start = time.time()
    for idx, file in enumerate(data_files):
        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)

        if idx == 0:
            predict_positions.append(center)
            tracker.new_init(image, center)
            start = time.time()
            if verbose:
                print("No.0 frame init position = {}".format(center))
        else:
            predict, similarity = tracker.new_track(image, verbose=verbose, frame_idx=idx)
            center = predict
            predict_positions.append(center)

    end = time.time()
    fps = (len(data_files) - 1) / (end - start)
    print("track speed: {} fps".format(fps))
    return predict_positions, fps


def load_student_model(model, pretrained_path, device):
    """Load the student model weights"""
    logger.info(f"Loading student model from {pretrained_path}")
    assert os.path.exists(pretrained_path), f"Model file not found: {pretrained_path}"

    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()
    matched = set(pretrained_dict.keys()) & set(model_dict.keys())
    missing = set(model_dict.keys()) - set(pretrained_dict.keys())
    logger.info(f"  Matched keys: {len(matched)}, Missing keys: {len(missing)}")

    filtered = {k: v for k, v in pretrained_dict.items() if k in set(model_dict.keys())}
    model.load_state_dict(filtered, strict=False)

    first_param = list(model.parameters())[0]
    logger.info(f"  First param norm: {first_param.data.norm().item():.6f}")
    return model


if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Load the student network (32³ version)
    student = MoCoStudentScorer32(in_channels=1)
    student = load_student_model(student, args.student_model, device)
    student = student.to(device).eval()
    logger.info(f"Student model loaded from {args.student_model}")

    # Set the seed (after model loading)
    set_seed(args.seed)

    # Create the tracker (32³ version)
    tracker = StudentMoCoTracker32(
        student_scorer=student,
        device=device,
        img_size=args.img_size,
        step=args.step,
        neighbor_type=args.neighbor_type
    )

    model_name = "moco_student_distilled_v3_32"
    logger.info(f"Tracker: {tracker.__class__.__name__}, "
                f"img_size={args.img_size}, neighbor={args.neighbor_type}")

    # Record the total time
    total_start = time.time()
    clust_track(args.clust_path, tracker, model_name, verbose=args.verbose)
    total_end = time.time()
    print("\nTotal test time: {:.2f} seconds".format(total_end - total_start))
