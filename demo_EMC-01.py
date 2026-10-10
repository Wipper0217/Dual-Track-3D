"""
EMC-01 single-series tracking demo (three-plane visualisation).

Based on clust_multi_branch_distilled_test.py, but restricted to the EMC-01
sequence and, in addition to the error numbers, it produces a three-plane
image for every annotated frame showing both the ground-truth annotation and
the tracker's prediction.

Pipeline (identical to the full test script):
  1. Load the 3D autoencoder (reg branch) and the distilled ResNet-18 student
     encoder (MoCo/dis branch).
  2. Build a MultiBranchTracker and track EMC-01 frame by frame.
  3. For each annotated frame, extract the axial / coronal / sagittal planes
     through the target and mark GT (red circle) and prediction (cyan cross).
  4. Save one PNG per annotated frame, an overview grid, and a metrics summary.

Usage:
    python demo_emc01_three_plane.py \
        --model_autoencoder snapshot/ResNetAutoEncoder3D_3d_100.pth \
        --model_distilled   snapshot/student_encoder_v3_best.pth \
        --clust_path        "D:/VS code/DATA/CLUST/clust_trainset_3d" \
        --cfg               experiments/moco/config_core_track.yaml \
        --out_dir           demo_output/EMC-01
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import importlib
import importlib.util
import logging
import random
import time
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import SimpleITK as sitk
import torch

from core.config_multi_branch import cfg
from utils.model_load import load_pretrain
from utils.metrics import error_metrics
from model.model_3d.resnet.resnet_ae_3d import ResNetAutoEncoder3D
from tracker.MultiBranchTracker_18 import MultiBranchTracker

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('demo_emc01')

# EMC series spacing (x, y, z) in mm/voxel, matching the main test script.
EMC_RESOLUTION = np.array([1.14, 0.59, 1.19])


# --------------------------------------------------------------------------- #
# Robust student-encoder loading
# --------------------------------------------------------------------------- #
def load_student_encoder_classes():
    """
    Return (StudentMoCoEncoder, StudentMoCoEncoder32).

    The knowledge-distillation package can be present under either the
    'knowledge_distill' or the legacy 'moco_test_new' name, and a stale
    __init__.py may break the normal package import. We therefore try the
    package import first and fall back to loading the module files directly.
    """
    for pkg in ("knowledge_distill", "moco_test_new"):
        try:
            m = importlib.import_module(f"{pkg}.model.student_encoder")
            m32 = importlib.import_module(f"{pkg}.model.student_encoder_32")
            return m.StudentMoCoEncoder, m32.StudentMoCoEncoder32
        except Exception:
            continue

    def _load_by_path(name, rel_path):
        path = os.path.join(PROJECT_ROOT, rel_path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    m = _load_by_path("_student_encoder", "knowledge_distill/model/student_encoder.py")
    m32 = _load_by_path("_student_encoder_32", "knowledge_distill/model/student_encoder_32.py")
    return m.StudentMoCoEncoder, m32.StudentMoCoEncoder32


def load_distilled_encoder(model, pretrained_path, device):
    """Load StudentMoCoEncoder weights from a MoCoStudentScorer checkpoint.

    Checkpoint keys look like 'encoder.conv1.weight'; the encoder model expects
    'conv1.weight', so the 'encoder.' prefix is stripped and the projection-head
    keys are dropped (only the encoder is needed at inference).
    """
    assert os.path.exists(pretrained_path), f"Model file not found: {pretrained_path}"
    logger.info(f"Loading distilled student encoder from {pretrained_path}")
    state = torch.load(pretrained_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if 'module.' in list(state.keys())[0]:
        state = {k.replace('module.', ''): v for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    filtered = {k[len('encoder.'):]: v for k, v in state.items()
                if k.startswith('encoder.') and k[len('encoder.'):] in model_keys}
    logger.info(f"  Matched keys: {len(filtered)}, Missing keys: {len(model_keys) - len(filtered)}")
    model.load_state_dict(filtered, strict=False)
    return model


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Tracking (mirrors model_track in the main test script)
# --------------------------------------------------------------------------- #
def track_series(data_files, tracker, init_center):
    """Track one sequence; returns predictions for every frame (frame 0 = init)."""
    predict_positions = []
    logger.info(f"Tracking {len(data_files)} frames ...")
    start = time.time()
    for idx, file in enumerate(data_files):
        image = sitk.GetArrayFromImage(sitk.ReadImage(file))
        if idx == 0:
            predict_positions.append(init_center)
            tracker.new_init(image, init_center)
        else:
            predict, _ = tracker.new_track(image)
            predict_positions.append(predict)
    elapsed = time.time() - start
    n = len(data_files) - 1
    logger.info(f"  done in {elapsed:.2f}s  ({n / elapsed:.1f} fps, {elapsed / n * 1000:.1f} ms/frame)")
    return predict_positions


# --------------------------------------------------------------------------- #
# Three-plane visualisation
# --------------------------------------------------------------------------- #
def extract_planes(volume, center):
    """Return the axial/coronal/sagittal slices through `center`=(cx,cy,cz).

    volume is a (D, H, W) array (z, y, x). The returned 2D arrays are oriented so
    that imshow's vertical axis is (axial) y, (coronal) z, (sagittal) z.
    """
    D, H, W = volume.shape
    cx, cy, cz = center
    x = int(np.clip(round(cx), 0, W - 1))
    y = int(np.clip(round(cy), 0, H - 1))
    z = int(np.clip(round(cz), 0, D - 1))
    return {
        'axial':    volume[z, :, :],   # (H, W) -> vertical y, horizontal x
        'coronal':  volume[:, y, :],   # (D, W) -> vertical z, horizontal x
        'sagittal': volume[:, :, x],   # (D, H) -> vertical z, horizontal y
    }, (x, y, z)


def _draw_markers(ax, plane, gt, pred):
    """Mark GT and prediction on one plane. Marker (h, v) = (horizontal, vertical)."""
    if plane == 'axial':       # horizontal x, vertical y
        gt_hv, pr_hv = (gt[0], gt[1]), (pred[0], pred[1])
    elif plane == 'coronal':   # horizontal x, vertical z
        gt_hv, pr_hv = (gt[0], gt[2]), (pred[0], pred[2])
    else:                      # sagittal: horizontal y, vertical z
        gt_hv, pr_hv = (gt[1], gt[2]), (pred[1], pred[2])

    ax.plot(gt_hv[0], gt_hv[1], marker='o', markersize=13, markerfacecolor='none',
            markeredgecolor='red', markeredgewidth=1.8, linestyle='none', zorder=5)
    ax.plot(pr_hv[0], pr_hv[1], marker='x', markersize=11, markeredgewidth=2.2,
            color='cyan', linestyle='none', zorder=6)


def save_three_plane(volume, gt_center, pred_center, frame_idx, error_mm,
                     spacing, out_path, zoom=45):
    """Save one three-plane figure for a single annotated frame."""
    planes, gt_xyz = extract_planes(volume, gt_center)
    _, pred_xyz = extract_planes(volume, pred_center)
    sx, sy, sz = spacing

    titles = {'axial': f'Axial (XY)  z={gt_xyz[2]}',
              'coronal': f'Coronal (XZ)  y={gt_xyz[1]}',
              'sagittal': f'Sagittal (YZ)  x={gt_xyz[0]}'}
    aspects = {'axial': sy / sx, 'coronal': sz / sx, 'sagittal': sz / sy}
    # (horizontal, vertical) marker coords per plane, for centring the zoom window
    ghv = {'axial': (gt_xyz[0], gt_xyz[1]), 'coronal': (gt_xyz[0], gt_xyz[2]),
           'sagittal': (gt_xyz[1], gt_xyz[2])}
    phv = {'axial': (pred_xyz[0], pred_xyz[1]), 'coronal': (pred_xyz[0], pred_xyz[2]),
           'sagittal': (pred_xyz[1], pred_xyz[2])}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, plane in enumerate(['axial', 'coronal', 'sagittal']):
        img = planes[plane]
        h, w = img.shape

        # Row 0: full plane
        ax0 = axes[0, col]
        ax0.imshow(img, cmap='gray', aspect=aspects[plane])
        _draw_markers(ax0, plane, gt_xyz, pred_xyz)
        ax0.set_title(titles[plane], fontsize=11)
        ax0.set_xticks([]); ax0.set_yticks([])

        # Row 1: zoomed crop centred between GT and prediction
        cu = (ghv[plane][0] + phv[plane][0]) / 2.0
        cv = (ghv[plane][1] + phv[plane][1]) / 2.0
        u0, u1 = int(cu) - zoom, int(cu) + zoom
        v0, v1 = int(cv) - zoom, int(cv) + zoom
        ax1 = axes[1, col]
        ax1.imshow(img, cmap='gray', aspect=aspects[plane])
        ax1.set_xlim(max(0, u0), min(w, u1))
        ax1.set_ylim(min(h, v1), max(0, v0))   # inverted y for image coords
        _draw_markers(ax1, plane, gt_xyz, pred_xyz)
        ax1.set_title(f'zoom +/-{zoom} vox', fontsize=9)
        ax1.set_xticks([]); ax1.set_yticks([])

    legend = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='red', markersize=12, markeredgewidth=1.8, label='Annotation (GT)'),
        Line2D([0], [0], marker='x', color='cyan', markersize=11,
               markeredgewidth=2.2, linestyle='none', label='Prediction'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=2, fontsize=11, frameon=False)
    fig.suptitle(
        f'EMC-01  frame {frame_idx + 1}  |  error {error_mm:.2f} mm  |  '
        f'GT=({gt_center[0]:.1f},{gt_center[1]:.1f},{gt_center[2]:.1f})  '
        f'Pred=({pred_center[0]:.1f},{pred_center[1]:.1f},{pred_center[2]:.1f})',
        fontsize=13, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_overview(items, out_path, zoom=45):
    """Grid: one row per annotated frame, three plane columns (zoomed)."""
    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for row, it in enumerate(items):
        planes, gt_xyz = extract_planes(it['volume'], it['gt'])
        _, pred_xyz = extract_planes(it['volume'], it['pred'])
        ghv = {'axial': (gt_xyz[0], gt_xyz[1]), 'coronal': (gt_xyz[0], gt_xyz[2]),
               'sagittal': (gt_xyz[1], gt_xyz[2])}
        phv = {'axial': (pred_xyz[0], pred_xyz[1]), 'coronal': (pred_xyz[0], pred_xyz[2]),
               'sagittal': (pred_xyz[1], pred_xyz[2])}
        for col, plane in enumerate(['axial', 'coronal', 'sagittal']):
            img = planes[plane]
            h, w = img.shape
            ax = axes[row, col]
            cu = (ghv[plane][0] + phv[plane][0]) / 2.0
            cv = (ghv[plane][1] + phv[plane][1]) / 2.0
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.set_xlim(max(0, int(cu) - zoom), min(w, int(cu) + zoom))
            ax.set_ylim(min(h, int(cv) + zoom), max(0, int(cv) - zoom))
            _draw_markers(ax, plane, gt_xyz, pred_xyz)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(plane.capitalize(), fontsize=11)
        axes[row, 0].set_ylabel(f"f{it['frame_idx'] + 1}\n{it['error_mm']:.2f} mm",
                                fontsize=10, rotation=0, ha='right', va='center', labelpad=34)
    fig.suptitle('EMC-01 annotated frames — GT (red o) vs prediction (cyan x)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='EMC-01 three-plane tracking demo')
    parser.add_argument('--cfg', type=str, default='experiments/moco/config_core_track.yaml')
    parser.add_argument('--model_autoencoder', type=str,
                        default='snapshot/ResNetAutoEncoder3D_3d_100.pth')
    parser.add_argument('--model_distilled', type=str,
                        default='snapshot/student_encoder_v3_best.pth')
    parser.add_argument('--clust_path', type=str,
                        default='D:/VS code/DATA/CLUST/clust_trainset_3d')
    parser.add_argument('--series', type=str, default='EMC-01')
    parser.add_argument('--target', type=int, default=1, help='which annotation file (1-based)')
    parser.add_argument('--out_dir', type=str, default='demo_output/EMC-01')
    parser.add_argument('--zoom', type=int, default=45, help='half-window of the zoomed crop (voxels)')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ----- config -----
    cfg.merge_from_file(args.cfg)
    reg_img_size = cfg.TRACKER.REG_IMG_SIZE
    dis_img_size = cfg.TRACKER.DIS_IMG_SIZE
    lambda_reg = cfg.TRACKER.LAMBDA_REG
    lambda_dis = cfg.TRACKER.LAMBDA_DIS

    # ----- models -----
    autoencoder = ResNetAutoEncoder3D(num_layers=18)
    autoencoder = load_pretrain(autoencoder, args.model_autoencoder).to(device)

    StudentMoCoEncoder, StudentMoCoEncoder32 = load_student_encoder_classes()
    if dis_img_size == 32:
        student = StudentMoCoEncoder32(in_channels=1)
    else:
        student = StudentMoCoEncoder(in_channels=1)
    student = load_distilled_encoder(student, args.model_distilled, device).to(device)
    student.eval()

    tracker = MultiBranchTracker(
        model_autoencoder=autoencoder, model_encoder=student,
        device=device, img_size=reg_img_size,
        reg_img_size=reg_img_size, dis_img_size=dis_img_size,
        lambda_reg=lambda_reg, lambda_dis=lambda_dis)
    logger.info(f"Tracker ready: reg_img_size={reg_img_size}, dis_img_size={dis_img_size}, "
                f"lambda_reg={lambda_reg}, lambda_dis={lambda_dis}")

    # ----- EMC-01 data -----
    series_path = os.path.join(args.clust_path, args.series)
    assert os.path.isdir(series_path), f"Series not found: {series_path}"
    data_files = sorted(glob(os.path.join(series_path, 'Data', '*.mha')))
    anno_file = os.path.join(series_path, 'Annotation', f'{args.series}_{args.target}.txt')
    assert data_files, f"No .mha files under {series_path}/Data"
    assert os.path.exists(anno_file), f"Annotation not found: {anno_file}"

    anno = np.loadtxt(anno_file)
    anno_frames = [int(f) for f in anno[:, 0]]         # 1-based frame indices
    anno_positions = anno[:, 1:]                       # (N, 3) as (cx, cy, cz)
    init_center = anno_positions[0].tolist()

    logger.info(f"{args.series}: {len(data_files)} frames, {len(anno_frames)} annotated frames "
                f"-> {anno_frames}")
    logger.info(f"Initial center: {initial_center_fmt(init_center)}")

    # ----- track -----
    predict_positions = track_series(data_files, tracker, init_center)

    # ----- build per-annotated-frame reports + figures -----
    os.makedirs(args.out_dir, exist_ok=True)
    items, rows = [], []
    for i, fidx in enumerate(anno_frames):
        frame0 = fidx - 1
        predict = np.asarray(predict_positions[frame0], dtype=float)
        gt = anno_positions[i]
        err_mm = float(np.linalg.norm((predict - gt) * EMC_RESOLUTION))

        volume = sitk.GetArrayFromImage(sitk.ReadImage(data_files[frame0]))
        out_png = os.path.join(args.out_dir, f'frame_{fidx:03d}_three_plane.png')
        save_three_plane(volume, gt, predict, frame0, err_mm,
                         EMC_RESOLUTION, out_png, zoom=args.zoom)
        logger.info(f"  frame {fidx:>3}: err={err_mm:6.2f} mm  -> {out_png}")

        items.append({'volume': volume, 'gt': gt, 'pred': predict,
                      'frame_idx': frame0, 'error_mm': err_mm})
        rows.append((fidx, gt, predict, err_mm))

    save_overview(items, os.path.join(args.out_dir, 'EMC-01_overview.png'), zoom=args.zoom)

    # ----- metrics + summary -----
    errors = np.array([r[3] for r in rows])
    m = error_metrics(errors, drop_first=True)   # drop frame 1 (init, error 0)
    m_all = error_metrics(errors, drop_first=False)

    summary_path = os.path.join(args.out_dir, 'EMC-01_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# EMC-01 tracking result (annotated frames)\n\n")
        f.write(f"- Series: `{args.series}`, target {args.target}\n")
        f.write(f"- Frames: {len(data_files)}, annotated frames: {len(anno_frames)}\n")
        f.write(f"- Resolution (x,y,z): {EMC_RESOLUTION.tolist()} mm/voxel\n")
        f.write(f"- Branch weights: lambda_reg={lambda_reg}, lambda_dis={lambda_dis}\n")
        f.write(f"- Models: {os.path.basename(args.model_autoencoder)} + "
                f"{os.path.basename(args.model_distilled)}\n\n")
        f.write("## Error per annotated frame\n\n")
        f.write("| frame | GT (cx,cy,cz) | prediction (cx,cy,cz) | error (mm) |\n")
        f.write("|-------|---------------|-----------------------|------------|\n")
        for fidx, gt, pred, err in rows:
            f.write(f"| {fidx} | ({gt[0]:.1f}, {gt[1]:.1f}, {gt[2]:.1f}) | "
                    f"({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}) | {err:.2f} |\n")
        f.write("\n## Summary\n\n")
        f.write(f"- All annotated frames: mean={m_all['mean']:.4f}, std={m_all['std']:.4f}, "
                f"p95={m_all['p95']:.4f}, max={m_all['max']:.4f} mm\n")
        f.write(f"- Excluding init frame 1: mean={m['mean']:.4f}, std={m['std']:.4f}, "
                f"p95={m['p95']:.4f}, max={m['max']:.4f} mm\n")

    # ----- persist predictions -----
    pred_path = os.path.join(args.out_dir, 'EMC-01_predictions.txt')
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write("# frame(cx,cy,cz) per tracked frame\n")
        for idx, p in enumerate(predict_positions):
            f.write(f"{idx + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    print("=" * 70)
    print(f"EMC-01 demo finished. {len(anno_frames)} annotated frames visualised.")
    print(f"  mean={m['mean']:.3f} mm, std={m['std']:.3f}, p95={m['p95']:.3f}, "
          f"max={m['max']:.3f} mm (excluding init frame)")
    print(f"  figures : {args.out_dir}")
    print(f"  summary : {summary_path}")
    print("=" * 70)


def initial_center_fmt(c):
    return f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"


if __name__ == '__main__':
    main()
