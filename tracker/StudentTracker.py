import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.transforms import Compose, ScaleIntensityd

from contrastive_learning.dataset.crop_utils import crop_numpy


class StudentTracker(nn.Module):
    """
    Tracker dedicated to the student network: single-branch, unified 32³ input, cosine similarity only
    """
    def __init__(self, student_encoder, device, img_size=32,
                 step=1, neighbor_type='18'):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.step = step
        self.student_encoder = student_encoder.to(device).eval()
        self.cache = {}

        # Pre-create the preprocessing transform (reused to avoid recreating it every time)
        self._transform = Compose([
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])

        # Cache the GPU volume of the current frame
        self._volume_gpu = None
        self._volume_shape = None
        self._volume_ptr = None

        if neighbor_type == '6':
            self._generate_neighbors = self._generate_6_neighbors
        elif neighbor_type == '18':
            self._generate_neighbors = self._generate_18_neighbors
        elif neighbor_type == '26':
            self._generate_neighbors = self._generate_26_neighbors
        else:
            self._generate_neighbors = self._generate_18_neighbors

    def _preprocess(self, image):
        data_dict = {"image": image}
        transform_img = self._transform(data_dict)
        return transform_img['image'].as_tensor().float()

    def _ensure_volume_gpu(self, volume):
        """Convert a numpy volume to a GPU tensor and normalize to [0, 1], with cache reuse"""
        if isinstance(volume, np.ndarray):
            volume_ptr = volume.ctypes.data
            if (self._volume_gpu is not None and 
                self._volume_ptr == volume_ptr and
                self._volume_shape == volume.shape):
                return self._volume_gpu

            tensor = torch.from_numpy(volume).float().to(self.device)
            v_min = tensor.min()
            v_max = tensor.max()
            tensor = (tensor - v_min) / (v_max - v_min + 1e-8)
            self._volume_gpu = tensor
            self._volume_shape = volume.shape
            self._volume_ptr = volume_ptr
            return self._volume_gpu
        elif isinstance(volume, torch.Tensor):
            if volume.device != self.device:
                volume = volume.to(self.device)
            if volume.dim() == 3:
                if volume.max() > 1.0 or volume.min() < 0.0:
                    v_min = volume.min()
                    v_max = volume.max()
                    volume = (volume - v_min) / (v_max - v_min + 1e-8)
                self._volume_gpu = volume
                self._volume_shape = volume.shape
            return self._volume_gpu
        else:
            raise TypeError(f"volume must be np.ndarray or torch.Tensor, got {type(volume)}")

    def new_init(self, img, center):
        """Initialize the template feature"""
        self.position = center
        z = crop_numpy(img, center, self.img_size)
        z = self._preprocess(z).to(self.device).unsqueeze(0).unsqueeze(0)
        self.z_feat = self.student_encoder.features(z)  # [1, 256]

    @torch.no_grad()
    def new_track(self, volume, step=None):
        if step is None:
            step = self.step

        p_0 = self.position
        self.cache.clear()

        while True:
            cx, cy, cz = p_0
            candidates = self._generate_neighbors([cx, cy, cz], step)

            # Check the cache and filter out the candidate points that need inference
            uncached_candidates = []
            uncached_indices = []
            similarities = []

            for idx, candidate in enumerate(candidates):
                key = self._generate_cache_key(candidate)
                cached_sim = self.cache.get(key, None)
                if cached_sim is not None:
                    similarities.append((idx, cached_sim))
                else:
                    uncached_candidates.append(candidate)
                    uncached_indices.append(idx)

            if len(uncached_candidates) > 0:
                # Batch-crop all uncached candidates
                crops = []
                for candidate in uncached_candidates:
                    x_crop = crop_numpy(volume, candidate, self.img_size)
                    x_crop = self._preprocess(x_crop).to(self.device).unsqueeze(0).unsqueeze(0)
                    crops.append(x_crop)
                
                # Batch forward pass
                batch = torch.cat(crops, dim=0)  # [N, 1, 32, 32, 32]
                x_feat_batch = self.student_encoder.features(batch)  # [N, 256]

                # Batch cosine similarity computation
                cos_sims = F.cosine_similarity(self.z_feat, x_feat_batch, dim=1)  # [N]

                for i, idx in enumerate(uncached_indices):
                    sim_val = cos_sims[i].item()
                    key = self._generate_cache_key(uncached_candidates[i])
                    self.cache[key] = sim_val
                    similarities.append((idx, sim_val))

            best_idx, best_sim = max(similarities, key=lambda x: x[1])
            p_max = candidates[best_idx]

            if self._find_best_position(p_0, p_max):
                break
            p_0 = p_max

        self.position = p_max
        return self.position, best_sim

    def _generate_cache_key(self, center):
        cx, cy, cz = center
        half = self.img_size // 2
        z0 = int(round(cz - half)); z1 = int(cz + half)
        y0 = int(round(cy - half)); y1 = int(cy + half)
        x0 = int(round(cx - half)); x1 = int(cx + half)
        return f"{z0}-{z1}_{y0}-{y1}_{x0}-{x1}"

    def _find_best_position(self, p_0, p_min):
        return p_0[0] == p_min[0] and p_0[1] == p_min[1] and p_0[2] == p_min[2]

    def _generate_6_neighbors(self, pos, step):
        cx, cy, cz = pos
        return [
            [cx, cy, cz],
            [cx + step, cy, cz], [cx - step, cy, cz],
            [cx, cy + step, cz], [cx, cy - step, cz],
            [cx, cy, cz + step], [cx, cy, cz - step]
        ]

    def _generate_18_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = []
        deltas = [0, step, -step]
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    if dx == 0 and dy == 0 and dz == 0:
                        candidates.append([cx, cy, cz])
                        continue
                    # 18-neighborhood: exclude the 8 corner points (all three directions non-zero)
                    if abs(dx) + abs(dy) + abs(dz) <= 2:
                        candidates.append([cx + dx, cy + dy, cz + dz])
        return candidates

    def _generate_26_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = []
        deltas = [0, step, -step]
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    candidates.append([cx + dx, cy + dy, cz + dz])
        return candidates
