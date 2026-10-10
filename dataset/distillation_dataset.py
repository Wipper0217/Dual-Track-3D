import json
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, ScaleIntensityd

from contrastive_learning.dataset.crop_utils import crop_numpy_safe


class DistillationTrackingDataset(Dataset):
    """
    Distillation training dataset
    Samples "template-candidate" pairs from annotated ultrasound sequences
    Each __getitem__ re-samples the candidate positions to increase data diversity
    """
    def __init__(self, root, json_path, label_path,
                 num_samples_per_frame=32,
                 pos_radius=2.0,
                 neg_radius_min=5.0,
                 img_size=32,
                 normalize=True):
        self.root = root
        self.img_size = img_size
        self.num_samples = num_samples_per_frame
        self.pos_radius = pos_radius
        self.neg_radius_min = neg_radius_min
        self.normalize = normalize

        # Load sequence metadata and annotations
        with open(json_path, 'r') as f:
            self.meta_data = json.load(f)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # Build the sample index list [(series, frame_idx, center_gt), ...]
        self.samples = self._build_samples()

        self._transform = Compose([
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])

    def _build_samples(self):
        samples = []
        for series_name, frame_indices in self.meta_data.items():
            if series_name not in self.labels:
                continue
            labels = self.labels[series_name]
            for i, frame_idx in enumerate(frame_indices):
                if i >= len(labels):
                    continue
                center_gt = labels[i]
                samples.append((series_name, frame_idx, center_gt))
        return samples

    def _load_frame(self, series_name, frame_idx):
        """Read the mha image for the specified series and frame index"""
        data_path = os.path.join(self.root, series_name, 'Data')
        data_files = sorted([f for f in os.listdir(data_path) if f.endswith('.mha')])
        # frame_idx is a 1-based frame number
        file_path = os.path.join(data_path, data_files[frame_idx - 1])
        img = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img)
        return img.astype(np.float32)

    def _preprocess(self, img):
        """Normalize and convert to a torch tensor"""
        if self.normalize:
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-6:
                img = (img - img_min) / (img_max - img_min)
        data_dict = {"image": img}
        transformed = self._transform(data_dict)
        tensor = transformed["image"].as_tensor().float()
        return tensor

    def _sample_candidates(self, center_gt):
        """
        Sample candidate positions (positive and negative samples), randomly generated on each call
        Increase sampling diversity
        """
        candidates = []
        pos_mask = []
        neg_mask = []

        # Positive samples: ground-truth position + small-range offset (with random perturbation)
        num_pos = self.num_samples // 2
        for _ in range(num_pos):
            # Add positional perturbation
            offset = np.random.uniform(-self.pos_radius, self.pos_radius, size=3)
            cand_center = np.array(center_gt) + offset
            candidates.append(cand_center)
            pos_mask.append(True)
            neg_mask.append(False)

        # Negative samples: random offset far from the ground-truth position (enlarge the random range)
        num_neg = self.num_samples - num_pos
        for _ in range(num_neg):
            # Random distance (enlarge the range to increase diversity)
            r = np.random.uniform(self.neg_radius_min, self.neg_radius_min + 10.0)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            offset = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            cand_center = np.array(center_gt) + offset
            candidates.append(cand_center)
            pos_mask.append(False)
            neg_mask.append(True)

        return candidates, pos_mask, neg_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        series, frame_idx, center_gt = self.samples[index]

        # Read the ultrasound image of this frame
        img = self._load_frame(series, frame_idx)

        # Crop the template centered on the ground-truth position
        template = crop_numpy_safe(img, center_gt, self.img_size)
        template = self._preprocess(template).unsqueeze(0)

        # Sample different candidate positions each time (increase data diversity)
        candidates, pos_mask, neg_mask = self._sample_candidates(center_gt)

        # Crop candidate image patches
        candidate_crops = []
        for cand in candidates:
            crop = crop_numpy_safe(img, cand, self.img_size)
            crop = self._preprocess(crop).unsqueeze(0)
            candidate_crops.append(crop)

        candidates_tensor = torch.stack(candidate_crops, dim=0)
        pos_mask_tensor = torch.tensor(pos_mask, dtype=torch.bool)
        neg_mask_tensor = torch.tensor(neg_mask, dtype=torch.bool)

        return {
            'template': template,
            'candidates': candidates_tensor,
            'pos_mask': pos_mask_tensor,
            'neg_mask': neg_mask_tensor,
            'center_gt': torch.tensor(center_gt, dtype=torch.float32)
        }
