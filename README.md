# Dual-Track-3D

Dual-branch 3D ultrasound tracking framework with knowledge distillation.

Each frame is tracked by combining two complementary similarity branches:

- **Autoencoder branch** — `ResNetAutoEncoder3D` features scored by cosine similarity.
- **MoCo branch** — self-supervised MoCo encoder features scored by NCC (normalized cross-correlation).

The two scores are combined with weights `LAMBDA_REG` (autoencoder) and `LAMBDA_DIS` (MoCo). A distilled ResNet-18 student can replace the heavy MoCo ResNet-50 teacher at inference for a substantial speedup with minimal accuracy loss.


## Requirements

Python 3.8+, PyTorch, NumPy, SimpleITK, MONAI, matplotlib, yacs.

## Training

```bash
# 3D autoencoder
python train_3d.py  --cfg experiments/auto-encoder/config_3d.yaml

# MoCo self-supervised pretraining
python main_moco.py --cfg experiments/moco/config_moco.yaml

# Distill the MoCo branch into a ResNet-18 student
python knowledge_distill/train_moco_distillation.py \
    --cfg knowledge_distill/experiments/config_moco_distillation.yaml
```

## Tracking / Evaluation

```bash
# Multi-branch tracking on the CLUST 3D dataset (teacher or distilled student)
python clust_multi_branch_distilled_test.py \
    --model_autoencoder snapshot/ResNetAutoEncoder3D_3d_100.pth \
    --model_distilled   snapshot/student_encoder_v3_best.pth
```
