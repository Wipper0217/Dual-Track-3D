"""
MoCo distillation dataset V3
Random-crop dataset for MoCo branch distillation

Sampling strategy:
- Randomly crop template-candidate pairs from 3D ultrasound images
- The candidate position is randomly offset around the template, simulating the tracking scenario
- Each sample contains a pair (template, candidate), both of size 64³
"""
import json
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.transforms import Compose, ScaleIntensityd


class MoCoDistillationDatasetV3(Dataset):
    """
    MoCo distillation training dataset V3

    Randomly crop template-candidate pairs from the CLUST 3D ultrasound dataset:
    1. Randomly select a sequence and frame
    2. Crop the template (64³) near the annotation position
    3. Crop the candidate (64³) with a random offset near the template
    4. Normalize the template and candidate separately

    Returns:
        template:  [1, 64, 64, 64] template image patch
        candidate: [1, 64, 64, 64] candidate image patch
    """
    def __init__(self, root, json_path, label_path,
                 img_size=64, student_img_size=None, normalize=True,
                 random_range=10, num_samples=512):
        """
        Args:
            root: data root directory
            json_path: sequence metadata JSON file path
            label_path: annotation JSON file path
            img_size: teacher crop size, default 64
            student_img_size: student crop size, default None (same as teacher)
            normalize: whether to normalize to [0, 1]
            random_range: random offset range of the candidate relative to the template (voxels)
            num_samples: number of samples per epoch
        """
        self.root = root
        self.img_size = img_size
        self.student_img_size = student_img_size if student_img_size else img_size
        self.normalize = normalize
        self.random_range = random_range
        self.num_samples = num_samples

        # Load sequence metadata and annotations
        with open(json_path, 'r') as f:
            self.meta_data = json.load(f)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # Build the series list
        self.series_list = self._build_series_list()

        # Shuffle and generate the sampling list
        self.pick = self._shuffle()

        # Normalization transform
        self._transform = Compose([
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])

    def _build_series_list(self):
        """Build the list of all available series"""
        series_list = []
        for series_name, frame_indices in self.meta_data.items():
            if series_name not in self.labels:
                continue
            labels = self.labels[series_name]
            series_list.append({
                'name': series_name,
                'frames': frame_indices,
                'labels': labels
            })
        return series_list

    def _shuffle(self):
        """Generate the random sampling list"""
        pick = []
        m = 0
        while m < self.num_samples:
            p = list(self.series_list)
            random.shuffle(p)
            pick.extend(p)
            m = len(pick)
        return pick[:self.num_samples]

    def _load_volume(self, series_name):
        """Load one mha file of a series"""
        data_path = os.path.join(self.root, series_name, 'Data')
        if not os.path.exists(data_path):
            return None
        data_files = sorted([f for f in os.listdir(data_path) if f.endswith('.mha')])
        if len(data_files) == 0:
            return None
        file_path = os.path.join(data_path, random.choice(data_files))
        img = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img)
        return img.astype(np.float32)

    def _random_crop(self, data, center, size):
        """Crop a patch from the data, handling boundary padding"""
        D, H, W = data.shape
        half = size // 2

        cx, cy, cz = center
        z0 = int(round(cz - half))
        y0 = int(round(cy - half))
        x0 = int(round(cx - half))
        z1 = z0 + size
        y1 = y0 + size
        x1 = x0 + size

        pad_z0 = max(0, -z0)
        pad_y0 = max(0, -y0)
        pad_x0 = max(0, -x0)
        pad_z1 = max(0, z1 - D)
        pad_y1 = max(0, y1 - H)
        pad_x1 = max(0, x1 - W)

        crop = data[max(0, z0):min(D, z1),
                    max(0, y0):min(H, y1),
                    max(0, x0):min(W, x1)]

        if any((pad_z0, pad_y0, pad_x0, pad_z1, pad_y1, pad_x1)):
            crop_tensor = torch.from_numpy(crop).float()
            crop_tensor = crop_tensor.unsqueeze(0).unsqueeze(0)
            crop_tensor = F.pad(
                crop_tensor,
                (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1),
                mode='constant', value=0.0
            )
            crop_tensor = crop_tensor.squeeze(0).squeeze(0)
            crop = crop_tensor.numpy()

        return crop

    def _get_random_center(self, data_shape, labels=None):
        """Get a random center point, preferentially sampled near annotation positions"""
        if labels is not None and len(labels) > 0:
            center = random.choice(labels)
            offset = np.random.uniform(-self.random_range, self.random_range, size=3)
            center = np.array(center) + offset
        else:
            D, H, W = data_shape
            cx = random.randint(D // 4, D * 3 // 4)
            cy = random.randint(H // 4, H * 3 // 4)
            cz = random.randint(W // 4, W * 3 // 4)
            center = np.array([cx, cy, cz])
        return center

    def _preprocess(self, img):
        """Normalize and convert to tensor"""
        if self.normalize:
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-6:
                img = (img - img_min) / (img_max - img_min)

        data_dict = {"image": img}
        transformed = self._transform(data_dict)
        tensor = transformed["image"].as_tensor().float()
        return tensor

    def __len__(self):
        return len(self.pick)

    def __getitem__(self, index):
        series_info = self.pick[index]
        series_name = series_info['name']
        labels = series_info['labels']

        # Load the volume data
        volume = self._load_volume(series_name)
        if volume is None:
            # Return random data on load failure (fault tolerance)
            template = torch.randn(1, self.img_size, self.img_size, self.img_size)
            candidate = torch.randn(1, self.img_size, self.img_size, self.img_size)
            result = {'template': template, 'candidate': candidate}
            if self.student_img_size != self.img_size:
                result['template_student'] = torch.randn(1, self.student_img_size, self.student_img_size, self.student_img_size)
                result['candidate_student'] = torch.randn(1, self.student_img_size, self.student_img_size, self.student_img_size)
            return result

        # Get the template center (selected from annotations)
        template_center = self._get_random_center(volume.shape, labels)

        # Get the candidate center (randomly offset near the template)
        offset = np.random.uniform(-self.random_range, self.random_range, size=3)
        candidate_center = template_center + offset

        # Crop the teacher template and candidate (64³)
        template_crop = self._random_crop(volume, template_center, self.img_size)
        candidate_crop = self._random_crop(volume, candidate_center, self.img_size)

        # Preprocess
        template_tensor = self._preprocess(template_crop).unsqueeze(0)  # [1, D, H, W]
        candidate_tensor = self._preprocess(candidate_crop).unsqueeze(0)  # [1, D, H, W]

        result = {
            'template': template_tensor,
            'candidate': candidate_tensor
        }

        # Crop the student template and candidate (32³)
        if self.student_img_size != self.img_size:
            template_s_crop = self._random_crop(volume, template_center, self.student_img_size)
            candidate_s_crop = self._random_crop(volume, candidate_center, self.student_img_size)
            result['template_student'] = self._preprocess(template_s_crop).unsqueeze(0)
            result['candidate_student'] = self._preprocess(candidate_s_crop).unsqueeze(0)

        return result
