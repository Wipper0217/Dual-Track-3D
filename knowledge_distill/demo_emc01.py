"""
EMC-01 single-sample tracking demo
Shows the complete tracking process of the student network on the EMC-01 sample in detail

Features:
1. Load the distilled student ResNet-18 model
2. Track the EMC-01 sequence frame by frame
3. Show the NCC score ranking of the 18 candidate points in each frame
4. Show the frame-by-frame displacement of the target position
5. Output an error comparison table against the ground truth

Usage:
    python moco_test_new/demo_emc01.py \
        --student_model snapshot/moco_distillation_v3/student_encoder_v3_best.pth
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import time
from glob import glob

import torch
import numpy as np
import SimpleITK as sitk

from moco_test_new.model.student_encoder import MoCoStudentScorer
from moco_test_new.tracker.student_moco_tracker import StudentMoCoTracker

parser = argparse.ArgumentParser(description='EMC-01 Single Sample Tracking Demo')
parser.add_argument('--student_model', type=str,
                    default="snapshot/moco_distillation_v3/student_encoder_v3_best.pth",
                    help='distilled student encoder model path')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--neighbor_type', type=str, default='18',
                    choices=['6', '18', '26'])
args = parser.parse_args()

# ============================================================
# EMC-01 configuration
# ============================================================
SERIES_PATH = "/root/data-pub/ultrasound/3DUS/CLUST/clust_trainset_3d/EMC-02"
RESOLUTION = np.array([1.14, 0.59, 1.19])  # EMC series resolution in mm/voxel


def load_student_model(model, pretrained_path, device):
    """Load the student model weights"""
    print(f"\n{'='*60}")
    print(f"Loading model from: {pretrained_path}")
    print(f"{'='*60}")

    assert os.path.exists(pretrained_path), f"Model not found: {pretrained_path}"
    pretrained_dict = torch.load(pretrained_path, map_location=device)

    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()
    matched = set(pretrained_dict.keys()) & set(model_dict.keys())
    missing = set(model_dict.keys()) - set(pretrained_dict.keys())

    print(f"  Matched keys: {len(matched)}")
    print(f"  Missing keys: {len(missing)}")

    filtered = {k: v for k, v in pretrained_dict.items() if k in set(model_dict.keys())}
    model.load_state_dict(filtered, strict=False)

    first_param = list(model.parameters())[0]
    print(f"  First param norm: {first_param.data.norm().item():.6f}")
    print(f"{'='*60}\n")

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load the student model
    student = MoCoStudentScorer(in_channels=1)
    student = load_student_model(student, args.student_model, device)
    student = student.to(device).eval()

    # 2. Create the tracker
    tracker = StudentMoCoTracker(
        student_scorer=student,
        device=device,
        img_size=args.img_size,
        step=args.step,
        neighbor_type=args.neighbor_type
    )

    # 3. Load the EMC-01 data
    data_path = os.path.join(SERIES_PATH, 'Data')
    anno_path = os.path.join(SERIES_PATH, 'Annotation')
    data_files = sorted(glob(os.path.join(data_path, '*.mha')))
    anno_file = os.path.join(anno_path, 'EMC-02_1.txt')
    anno = np.loadtxt(anno_file)

    anno_frames = anno[:, 0].tolist()
    anno_positions = anno[:, 1:]

    initial_center = anno_positions[0].tolist()

    print(f"\n{'#'*60}")
    print(f"# EMC-01 Tracking Demo")
    print(f"{'#'*60}")
    print(f"  Series path:      {SERIES_PATH}")
    print(f"  Total frames:     {len(data_files)}")
    print(f"  Annotated frames: {len(anno_frames)}")
    print(f"  Resolution:       {RESOLUTION} mm/voxel")
    print(f"  Initial position: ({initial_center[0]:.1f}, {initial_center[1]:.1f}, {initial_center[2]:.1f})")
    print(f"  Neighbor type:    {args.neighbor_type}")
    print(f"  Search step:      {args.step}")
    print(f"{'#'*60}\n")

    # 4. Frame-by-frame tracking
    predict_positions = []
    frame_times = []

    for idx, file in enumerate(data_files):
        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)

        if idx == 0:
            # First frame: initialization
            predict_positions.append(initial_center)
            tracker.new_init(image, initial_center)
            print(f"{'='*60}")
            print(f"  [Frame 0] INITIALIZATION")
            print(f"  Position: ({initial_center[0]:.1f}, {initial_center[1]:.1f}, {initial_center[2]:.1f})")
            print(f"{'='*60}")
        else:
            # Subsequent frames: tracking
            start = time.time()
            predict, similarity = tracker.new_track(
                image, verbose=True, frame_idx=idx
            )
            end = time.time()
            elapsed = end - start
            frame_times.append(elapsed)

            prev_pos = predict_positions[-1]
            displacement = np.sqrt(
                (predict[0] - prev_pos[0])**2 +
                (predict[1] - prev_pos[1])**2 +
                (predict[2] - prev_pos[2])**2
            )

            print(f"  [Frame {idx}] Result: pos=({predict[0]:.1f}, {predict[1]:.1f}, {predict[2]:.1f})  "
                  f"NCC={similarity:+.6f}  disp={displacement:.1f}vox  time={elapsed*1000:.1f}ms")

            predict_positions.append(predict)

    # 5. Tracking result summary
    print(f"\n\n{'='*60}")
    print(f"  EMC-01 TRACKING RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Frame':>5s}  {'Pred X':>8s} {'Pred Y':>8s} {'Pred Z':>8s}  "
          f"{'GT X':>8s} {'GT Y':>8s} {'GT Z':>8s}  {'Err(mm)':>8s}")
    print(f"  {'-'*5}  {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    errors = []
    for i in range(len(anno_frames)):
        frame_idx = int(anno_frames[i]) - 1
        predict = predict_positions[frame_idx]
        label = anno_positions[i]

        error_vec = predict - label
        error_mm = error_vec * RESOLUTION
        error_dist = np.sqrt(np.sum(error_mm**2))
        errors.append(error_dist)

        print(f"  {frame_idx+1:5d}  "
              f"{predict[0]:8.2f} {predict[1]:8.2f} {predict[2]:8.2f}  "
              f"{label[0]:8.2f} {label[1]:8.2f} {label[2]:8.2f}  "
              f"{error_dist:8.2f}")

    # 6. Statistics
    errors = np.array(errors)
    print(f"  {'-'*5}  {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")
    print(f"\n  Statistics:")
    print(f"    Mean Error:    {errors.mean():.2f} mm")
    print(f"    Std Error:     {errors.std():.2f} mm")
    print(f"    Median Error:  {np.median(errors):.2f} mm")
    print(f"    Max Error:     {errors.max():.2f} mm")
    print(f"    Min Error:     {errors.min():.2f} mm")
    print(f"    95th TE:       {np.percentile(errors, 95):.2f} mm")

    if len(frame_times) > 0:
        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time
        print(f"\n  Speed:")
        print(f"    Avg frame time: {avg_time*1000:.1f} ms")
        print(f"    FPS:            {fps:.1f}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
