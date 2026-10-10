import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.transforms import Compose, ScaleIntensityd

from contrastive_learning.dataset.crop_utils import crop_and_resize_numpy, crop_numpy

class MultiBranchTracker(nn.Module):
    def __init__(self, model_autoencoder, model_encoder, device, img_size,
                 reg_img_size=32, dis_img_size=64,
                 lambda_reg=0.2, lambda_dis=0.8):
        super(MultiBranchTracker, self).__init__()
        self.device = device
        self.img_size = img_size

        self.reg_img_size = reg_img_size
        self.dis_img_size = dis_img_size
        self.lambda_reg = lambda_reg
        self.lambda_dis = lambda_dis

        self.cache = dict()
        self._set_model(model_autoencoder, model_encoder)

        # Pre-create the preprocessing transform
        self._transform = Compose([
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])

        # Cache the GPU volume of the current frame
        self._volume_gpu = None
        self._volume_shape = None
        self._volume_ptr = None  # Used to detect whether the numpy array is still the same data

    def _set_model(self, model_autoencoder, model_encoder):
        self.model_autoencoder = model_autoencoder.to(self.device)
        self.model_encoder = model_encoder.to(self.device)
        self.model_autoencoder.eval()
        self.model_encoder.eval()

    def generate_cache_key(self, center):
        cx, cy, cz = center
        half_size = self.img_size // 2

        z0 = int(round(cz - half_size))
        y0 = int(round(cy - half_size))
        x0 = int(round(cx - half_size))

        z1 = int(cz + half_size)
        y1 = int(cy + half_size)
        x1 = int(cx + half_size)

        key = str(z0) + '-' + str(z1) + '_' + str(y0) + '-' + str(y1) + '_' + str(x0) + '-' + str(x1)
        return key

    def get_name(self):
        return "MultiBranchTracker"

    def pre_process(self, image):
        data_dict = {
            "image": image
        }
        transform_img = self._transform(data_dict)
        return transform_img['image'].as_tensor().float()

    def _ensure_volume_gpu(self, volume, force_update=False):
        """Convert a numpy volume to a GPU tensor and normalize to [0, 1], with cache reuse

        Note: do not use the memory address (ctypes.data) to decide caching,
        because numpy reuses the memory address for arrays of the same shape in a loop,
        which causes volumes from different frames to be misidentified as the same one,
        returning stale data.
        Every new frame must be forced to update via force_update=True.
        """
        if isinstance(volume, np.ndarray):
            if force_update or self._volume_gpu is None or self._volume_shape != volume.shape:
                tensor = torch.from_numpy(volume).float().to(self.device)
                # Normalize the whole frame
                v_min = tensor.min()
                v_max = tensor.max()
                tensor = (tensor - v_min) / (v_max - v_min + 1e-8)
                self._volume_gpu = tensor  # (D, H, W)
                self._volume_shape = volume.shape
            return self._volume_gpu
        elif isinstance(volume, torch.Tensor):
            if volume.device != self.device:
                volume = volume.to(self.device)
            if volume.dim() == 3:
                # Normalize if not already normalized
                if volume.max() > 1.0 or volume.min() < 0.0:
                    v_min = volume.min()
                    v_max = volume.max()
                    volume = (volume - v_min) / (v_max - v_min + 1e-8)
                self._volume_gpu = volume
                self._volume_shape = volume.shape
            return self._volume_gpu
        else:
            raise TypeError(f"volume must be np.ndarray or torch.Tensor, got {type(volume)}")

    def _normalize_patches(self, patches):
        """
        Apply min-max normalization to [0, 1] for each patch in the batch separately
        patches: [N, 1, D, H, W]
        """
        N = patches.shape[0]
        for i in range(N):
            patch = patches[i]
            p_min = patch.min()
            p_max = patch.max()
            patches[i] = (patch - p_min) / (p_max - p_min + 1e-8)
        return patches

    def _crop_gpu_batch(self, volume_gpu, centers, output_size):
        """Batch-crop directly on a GPU tensor

        :param volume_gpu: (D, H, W) GPU tensor, already normalized
        :param centers: list of [cx, cy, cz]
        :param output_size: int
        :return: (N, 1, output_size, output_size, output_size) GPU tensor
        """
        D, H, W = volume_gpu.shape
        half = output_size // 2
        patches = []

        for center in centers:
            cx, cy, cz = center
            z0 = int(round(cz - half))
            y0 = int(round(cy - half))
            x0 = int(round(cx - half))
            z1 = z0 + output_size
            y1 = y0 + output_size
            x1 = x0 + output_size

            # Boundary check and padding (using pad + slice on the GPU)
            pad_z0 = max(0, -z0)
            pad_y0 = max(0, -y0)
            pad_x0 = max(0, -x0)
            pad_z1 = max(0, z1 - D)
            pad_y1 = max(0, y1 - H)
            pad_x1 = max(0, x1 - W)

            if any((pad_z0, pad_y0, pad_x0, pad_z1, pad_y1, pad_x1)):
                patch = volume_gpu[
                    max(0, z0):min(D, z1),
                    max(0, y0):min(H, y1),
                    max(0, x0):min(W, x1)
                ]
                # F.pad format: (last dim left, last dim right, ..., first dim left, first dim right)
                patch = F.pad(
                    patch.unsqueeze(0).unsqueeze(0),
                    (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1),
                    mode='constant',
                    value=0.0
                ).squeeze(0).squeeze(0)
            else:
                patch = volume_gpu[z0:z1, y0:y1, x0:x1]

            patches.append(patch)

        batch = torch.stack(patches, dim=0).unsqueeze(1)  # (N, 1, D, H, W)
        return batch

    def new_init(self, img, center):
        self.position = center
        # Force-update the volume every frame (the first frame of a new sequence/new target cannot use the cache)
        volume_gpu = self._ensure_volume_gpu(img, force_update=True)

        z_reg = self._crop_gpu_batch(volume_gpu, [center], self.reg_img_size)  # (1, 1, D, H, W)
        z_dis = self._crop_gpu_batch(volume_gpu, [center], self.dis_img_size)
        z_reg = self._normalize_patches(z_reg)
        z_dis = self._normalize_patches(z_dis)

        self.z1 = self.model_autoencoder.features(z_reg, False)
        self.z2 = self.model_encoder(z_dis)

    def init(self, img, center):
        self.position = center
        volume_gpu = self._ensure_volume_gpu(img, force_update=True)

        z = self._crop_gpu_batch(volume_gpu, [center], self.img_size)
        z = self._normalize_patches(z)

        self.z1 = self.model_autoencoder.features(z, False)
        self.z2 = self.model_encoder(z)

    def _init_search_cache(self):
        self.cache = {}

    @torch.no_grad()
    def track(self, volume, step=1):
        return self.status_cache_search(volume, step)

    @torch.no_grad()
    def new_track(self, volume, step=1):
        """Batch-inference + in-GPU-crop version of new_track"""
        # The volume must be force-updated every frame (different frames have the same shape, so shape cannot determine whether an update is needed)
        volume_gpu = self._ensure_volume_gpu(volume, force_update=True)
        p_0 = self.position
        self.cache.clear()

        while True:
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)

            # Check the cache and filter out the candidate points that need inference
            uncached_candidates = []
            uncached_indices = []
            similarities = []

            for idx, candidate in enumerate(candidates):
                key = self.generate_cache_key(candidate)
                cached_sim = self.cache.get(key, None)
                if cached_sim is not None:
                    similarities.append((idx, cached_sim))
                else:
                    uncached_candidates.append(candidate)
                    uncached_indices.append(idx)

            if len(uncached_candidates) > 0:
                # In-GPU crop for the reg branch
                batch_reg = self._crop_gpu_batch(volume_gpu, uncached_candidates, self.reg_img_size)
                # In-GPU crop for the dis branch
                batch_dis = self._crop_gpu_batch(volume_gpu, uncached_candidates, self.dis_img_size)
                batch_reg = self._normalize_patches(batch_reg)
                batch_dis = self._normalize_patches(batch_dis)

                # Batch forward pass
                x1_batch = self.model_autoencoder.features(batch_reg, False)
                x2_batch = self.model_encoder(batch_dis)

                # Batch similarity computation
                batch_similarities = self.calculate_similarity_batch(
                    x1_batch, x2_batch,
                    lambda_1=self.lambda_reg,
                    lambda_2=self.lambda_dis
                )

                for i, idx in enumerate(uncached_indices):
                    sim_val = batch_similarities[i].item()
                    key = self.generate_cache_key(uncached_candidates[i])
                    self.cache[key] = sim_val
                    similarities.append((idx, sim_val))

            best_idx, best_sim = max(similarities, key=lambda x: x[1])
            p_max = candidates[best_idx]

            if self.find_best_position(p_0, p_max):
                break
            p_0 = p_max

        self.position = p_max
        return self.position, best_sim

    @torch.no_grad()
    def status_cache_search(self, volume, step=1):
        """Batch-inference + in-GPU-crop version of status_cache_search"""
        volume_gpu = self._ensure_volume_gpu(volume, force_update=True)
        p_0 = self.position
        self.cache.clear()

        while True:
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)

            uncached_candidates = []
            uncached_indices = []
            similarities = []

            for idx, candidate in enumerate(candidates):
                key = self.generate_cache_key(candidate)
                cached_sim = self.cache.get(key, None)
                if cached_sim is not None:
                    similarities.append((idx, cached_sim))
                else:
                    uncached_candidates.append(candidate)
                    uncached_indices.append(idx)

            if len(uncached_candidates) > 0:
                batch_imgs = self._crop_gpu_batch(volume_gpu, uncached_candidates, self.img_size)
                batch_imgs = self._normalize_patches(batch_imgs)

                x1_batch = self.model_autoencoder.features(batch_imgs, False)
                x2_batch = self.model_encoder(batch_imgs)

                batch_similarities = self.calculate_similarity_batch(
                    x1_batch, x2_batch,
                    lambda_1=0.2,
                    lambda_2=0.8
                )

                for i, idx in enumerate(uncached_indices):
                    sim_val = batch_similarities[i].item()
                    key = self.generate_cache_key(uncached_candidates[i])
                    self.cache[key] = sim_val
                    similarities.append((idx, sim_val))

            best_idx, best_sim = max(similarities, key=lambda x: x[1])
            p_max = candidates[best_idx]

            if self.find_best_position(p_0, p_max):
                break
            p_0 = p_max

        self.position = p_max
        return self.position, best_sim

    def calculate_similarity(self, x1, x2, lambda_1=1.0, lambda_2=1.0):
        ae_sim = self.calculate_cosine_similarity(self.z1, x1)
        e_sim = self.calculate_ncc(self.z2, x2)
        return lambda_1 * ae_sim + lambda_2 * e_sim

    def calculate_similarity_batch(self, x1_batch, x2_batch, lambda_1=1.0, lambda_2=1.0):
        ae_sim = self.calculate_cosine_similarity_batch(self.z1, x1_batch)
        e_sim = self.calculate_ncc_batch(self.z2, x2_batch)
        return lambda_1 * ae_sim + lambda_2 * e_sim

    def calculate_cosine_similarity(self, z, x):
        z = z.view(1, -1)
        x = x.view(1, -1)
        cos_sim = F.cosine_similarity(z, x, dim=1)
        return cos_sim

    def calculate_cosine_similarity_batch(self, z, x_batch):
        N = x_batch.shape[0]
        # Handle z
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, C)
        elif z.dim() >= 2:
            z = z.view(1, -1)  # (1, C)
        # Handle x_batch
        if x_batch.dim() >= 2:
            x_batch = x_batch.view(N, -1)  # (N, C)
        z_expanded = z.expand(N, -1)
        cos_sim = F.cosine_similarity(z_expanded, x_batch, dim=1)
        return cos_sim

    def calculate_relative_similarity(self, z, x):
        sim_x = self.calculate_cosine_similarity(z, x)

    def calculate_ncc(self, z, x, eps=1e-8):
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            z = z.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(0)
            z = z.unsqueeze(0)

        B, C, D, H, W = z.shape
        z_flat = z.view(B, C, -1)
        x_flat = x.view(B, C, -1)

        x_mean = x_flat.mean(dim=2, keepdim=True)
        z_mean = z_flat.mean(dim=2, keepdim=True)

        x_centered = x_flat - x_mean
        z_centered = z_flat - z_mean

        cov = (x_centered * z_centered).sum(dim=2)

        x_var = (x_centered ** 2).sum(dim=2)
        z_var = (z_centered ** 2).sum(dim=2)
        std_prod = torch.sqrt(x_var * z_var + eps)

        ncc_per_channel = cov / std_prod

        ncc = ncc_per_channel.mean(dim=1)

        if B == 1:
            return ncc[0].item()
        else:
            return ncc

    def calculate_ncc_batch(self, z, x_batch, eps=1e-8):
        """Batch NCC computation, handling the case where the encoder output is (B,C,1,1,1)

        z: template feature, may be (B,C,1,1,1) or (B,C) or (C,)
        x_batch: (N, C, ...) candidate features
        returns: (N,) tensor
        """
        N = x_batch.shape[0]

        # Detect whether z was pooled by AdaptiveAvgPool down to (1,1,1)
        z_is_pooled = z.dim() == 5 and z.shape[2:] == (1, 1, 1)
        z_is_pooled = z_is_pooled or (z.dim() == 4 and z.shape[2:] == (1, 1))
        z_is_pooled = z_is_pooled or (z.dim() == 3 and z.shape[1:] == (1, 1))

        if z_is_pooled:
            # z is the avg pool output (B,C,1,1,1) -> flatten to (B,C)
            if z.dim() >= 3:
                z_flat = z.view(z.shape[0], z.shape[1])  # (1, C)
            else:
                z_flat = z
        else:
            # z has spatial dimensions, flatten
            if z.dim() >= 3:
                z_flat = z.view(z.shape[0], z.shape[1], -1)  # (1, C, L)
            else:
                z_flat = z

        # Flatten x_batch
        if x_batch.dim() == 5:
            x_flat = x_batch.view(N, x_batch.shape[1], -1)  # (N, C, L)
        elif x_batch.dim() == 4:
            x_flat = x_batch.view(N, x_batch.shape[1], -1)
        elif x_batch.dim() == 3:
            x_flat = x_batch.unsqueeze(1).view(N, 1, -1)
        elif x_batch.dim() == 2:
            x_flat = x_batch.unsqueeze(2)
        else:
            raise ValueError(f"x_batch dim {x_batch.dim()} not supported")

        # z_flat is (B,C) or (B,C,L), x_flat is (N,C,L)
        # Unify to (N, C) vector NCC
        if z_flat.dim() == 2 and x_flat.dim() == 3:
            # z_flat: (1, C), x_flat: (N, C, L) -> flatten x_flat to (N, C*L)
            x_flat = x_flat.view(N, -1)  # (N, C*L)
            z_flat = z_flat.view(z_flat.shape[0], -1).expand(N, -1)  # (N, C*L)
        elif z_flat.dim() == 3 and x_flat.dim() == 3:
            # Both have spatial dimensions; compute after flattening
            x_flat = x_flat.view(N, -1)
            z_flat = z_flat.view(z_flat.shape[0], -1).expand(N, -1)
        elif z_flat.dim() == 2 and x_flat.dim() == 2:
            z_flat = z_flat.expand(N, -1)

        x_mean = x_flat.mean(dim=1, keepdim=True)
        z_mean = z_flat.mean(dim=1, keepdim=True)

        x_centered = x_flat - x_mean
        z_centered = z_flat - z_mean

        cov = (x_centered * z_centered).sum(dim=1)

        x_var = (x_centered ** 2).sum(dim=1)
        z_var = (z_centered ** 2).sum(dim=1)
        std_prod = torch.sqrt(x_var * z_var + eps)

        ncc = cov / std_prod  # (N,)
        return ncc

    def _generate_6_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = [[cx, cy, cz]]
        candidates.extend([
            [cx + step, cy, cz],
            [cx - step, cy, cz],
            [cx, cy + step, cz],
            [cx, cy - step, cz],
            [cx, cy, cz + step],
            [cx, cy, cz - step],
        ])
        return candidates

    def _generate_18_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = self._generate_6_neighbors(pos, step)
        candidates.extend([
            [cx + step, cy + step, cz],
            [cx + step, cy - step, cz],
            [cx - step, cy + step, cz],
            [cx - step, cy - step, cz],
            [cx + step, cy, cz + step],
            [cx + step, cy, cz - step],
            [cx - step, cy, cz + step],
            [cx - step, cy, cz - step],
            [cx, cy + step, cz + step],
            [cx, cy + step, cz - step],
            [cx, cy - step, cz + step],
            [cx, cy - step, cz - step],
        ])
        return candidates

    def _generate_26_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = self._generate_18_neighbors(pos, step)
        candidates.extend([
            [cx + step, cy + step, cz + step],
            [cx + step, cy + step, cz - step],
            [cx + step, cy - step, cz + step],
            [cx + step, cy - step, cz - step],
            [cx - step, cy + step, cz + step],
            [cx - step, cy + step, cz - step],
            [cx - step, cy - step, cz + step],
            [cx - step, cy - step, cz - step],
        ])
        return candidates

    def _generate_search_region(self, pos, step):
        return self._generate_26_neighbors(pos, step)

    def find_best_position(self, p_0, p_min):
        if p_0[0] == p_min[0] and p_0[1] == p_min[1] and p_0[2] == p_min[2]:
            return True
        return False

    @torch.no_grad()
    def my_track(self, volume, search_radius=1, next_search_amount=5):
        next_search_centers = self.coarse_search(volume, search_radius, next_search_amount)
        predict_pos = self.fine_search(volume, next_search_centers)
        self.position = predict_pos
        return predict_pos

    def coarse_search(self, volume, search_radius, next_search_amount):
        last_pos = self.position
        candidates_pos = self._generate_search_region(last_pos, search_radius)
        volume_gpu = self._ensure_volume_gpu(volume, force_update=True)

        batch_imgs = self._crop_gpu_batch(volume_gpu, candidates_pos, self.img_size)
        batch_imgs = self._normalize_patches(batch_imgs)

        x_batch = self.model_encoder(batch_imgs)

        sims = self.calculate_cosine_similarity_batch(self.z2, x_batch)

        next_search_centers = []
        for i, candidate in enumerate(candidates_pos):
            next_search_centers.append([candidate, sims[i].item()])

        next_search_centers = sorted(next_search_centers, key=lambda x: x[1], reverse=True)
        return next_search_centers[:next_search_amount]

    def _generate_key(self, pos):
        cx, cy, cz = map(round, pos)
        key = 'pos_' + str(cx) + '_' + str(cy) + '_' + str(cz)
        return key

    def fine_search(self, volume, next_search_centers):
        candidate_predict = []
        for candidate, distance in next_search_centers:
            p_min, similarity = self._greedy_search(volume, candidate, 1)
            candidate_predict.append([p_min, similarity])

        candidate_predict = sorted(candidate_predict, key=lambda x: x[1], reverse=True)
        return candidate_predict[0][0]

    def _greedy_search(self, volume, pos, step):
        p_0 = pos
        volume_gpu = self._ensure_volume_gpu(volume, force_update=True)
        while True:
            similarity = float('-inf')
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)

            batch_imgs = self._crop_gpu_batch(volume_gpu, candidates, self.img_size)
            batch_imgs = self._normalize_patches(batch_imgs)

            x1_batch = self.model_autoencoder.features(batch_imgs, False)
            x2_batch = self.model_encoder(batch_imgs)

            batch_sims = self.calculate_similarity_batch(x1_batch, x2_batch)

            best_idx = torch.argmax(batch_sims).item()
            best_sim = batch_sims[best_idx].item()
            p_min = candidates[best_idx]

            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        return p_min, best_sim
