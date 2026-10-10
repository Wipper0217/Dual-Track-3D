import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OptimizedMultiBranchTracker(nn.Module):
    """Optimized dual-branch 3D tracker with batch GPU inference.

    Replaces the per-candidate loop in MultiBranchTracker_18.py with batch
    processing of all 27 candidates (center + 26 neighbors) in a single
    forward pass, yielding 3-5x speedup.

    Design decisions (matching MultiBranchTracker_18 behavior):
    - No global volume normalization: raw volume transferred to GPU as float32,
      per-patch min-max normalization applied after cropping.
    - Pure PyTorch preprocessing (no MONAI dependency).
    - AE branch uses reg_img_size=32, MoCo branch uses dis_img_size=64.
    - Cache-based deduplication of candidate positions across greedy iterations.
    """

    def __init__(self, model_autoencoder, model_encoder, device,
                 reg_img_size=32, dis_img_size=64,
                 lambda_reg=1.0, lambda_dis=1.0,
                 use_amp=False, use_compile=False,
                 use_adaptive_fusion=False, fusion_temperature=1.0,
                 top_k_fusion=0, top_k_weight_ae=0.5,
                 use_weighted_ncc=False):
        """Initialize the optimized dual-branch tracker.

        Args:
            top_k_fusion: If >0, use Top-K fusion: AE scores all candidates,
                then MoCo re-ranks only the top-K AE candidates. Z-score
                normalized scores from both branches are combined. 0 = disabled.
            top_k_weight_ae: Weight for AE in Top-K fusion (0-1).
                Higher = more trust in AE, lower = more trust in MoCo.
            use_weighted_ncc: If True, use variance-based channel weighting
                for per-channel NCC computation.
        """
        super(OptimizedMultiBranchTracker, self).__init__()
        self.device = device
        self.reg_img_size = reg_img_size
        self.dis_img_size = dis_img_size
        self.lambda_reg = lambda_reg
        self.lambda_dis = lambda_dis
        self.use_amp = use_amp
        self.use_adaptive_fusion = use_adaptive_fusion
        self.fusion_temperature = fusion_temperature
        self.top_k_fusion = top_k_fusion
        self.top_k_weight_ae = top_k_weight_ae
        self.use_weighted_ncc = use_weighted_ncc

        # img_size used for cache key generation
        # Use max crop size to ensure cache key covers the largest patch boundary
        self.img_size = max(reg_img_size, dis_img_size)

        self.cache = dict()
        self.position = None
        self.z1 = None
        self.z2 = None

        # Pre-allocated GPU volume buffer (reused across frames)
        self._volume_buffer = None
        self._buffer_shape = None

        self._set_model(model_autoencoder, model_encoder, use_compile)

    def _set_model(self, model_autoencoder, model_encoder, use_compile=False):
        self.model_autoencoder = model_autoencoder.to(self.device)
        self.model_encoder = model_encoder.to(self.device)
        self.model_autoencoder.eval()
        self.model_encoder.eval()

        if use_compile and hasattr(torch, 'compile'):
            self.model_autoencoder = torch.compile(self.model_autoencoder, mode='reduce-overhead')
            self.model_encoder = torch.compile(self.model_encoder, mode='reduce-overhead')

    def _warmup_compile(self):
        """Warm up torch.compile with a dummy forward pass to avoid first-frame JIT overhead."""
        dummy_reg = torch.randn(1, 1, self.reg_img_size, self.reg_img_size, self.reg_img_size,
                                device=self.device, dtype=torch.float32)
        dummy_dis = torch.randn(1, 1, self.dis_img_size, self.dis_img_size, self.dis_img_size,
                                device=self.device, dtype=torch.float32)
        with torch.no_grad():
            _ = self.model_autoencoder.features(dummy_reg, False)
            _ = self.model_encoder(dummy_dis)

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
        return "OptimizedMultiBranchTracker"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _volume_to_gpu(self, volume):
        """Transfer numpy volume to GPU with buffer reuse.

        Avoids repeated allocation by reusing a pre-allocated GPU buffer
        when the volume shape matches the previous frame.

        Args:
            volume: 3D numpy array (D, H, W).

        Returns:
            (D, H, W) GPU float32 tensor.
        """
        vol_np = np.asarray(volume, dtype=np.float32)
        shape = vol_np.shape

        if (self._volume_buffer is not None and
                self._buffer_shape == shape):
            # Reuse existing buffer — just copy new data
            self._volume_buffer.copy_(torch.from_numpy(vol_np))
            return self._volume_buffer
        else:
            # Allocate new buffer
            self._volume_buffer = torch.from_numpy(vol_np).to(self.device)
            self._buffer_shape = shape
            return self._volume_buffer

    @staticmethod
    def _normalize_patches(patches):
        """Vectorized per-patch min-max normalization to [0, 1].

        Equivalent to MONAI ScaleIntensityd(minv=0, maxv=1) applied to each
        patch independently. Operates on the full batch at once using
        view-based min/max (no Python loop over patches).

        Args:
            patches: [N, 1, D, H, W] tensor (GPU or CPU).

        Returns:
            [N, 1, D, H, W] tensor with each patch in [0, 1].
        """
        N = patches.shape[0]
        flat = patches.view(N, -1)                    # (N, D*H*W)
        p_min = flat.min(dim=1, keepdim=True).values   # (N, 1)
        p_max = flat.max(dim=1, keepdim=True).values   # (N, 1)
        flat = (flat - p_min) / (p_max - p_min + 1e-8)
        return flat.view_as(patches)

    # ------------------------------------------------------------------
    # GPU batch crop
    # ------------------------------------------------------------------

    def _crop_gpu_batch(self, volume_gpu, centers, output_size):
        """Batch crop from a GPU volume tensor with boundary zero-padding.

        Mirrors crop_numpy(center, output_size) + np.pad for boundary cases.
        Uses int(round(...)) for both start and end coordinates (matching
        crop_numpy's start convention; end uses int() in the original but
        round() here for symmetry -- equivalent for integer centers).

        Args:
            volume_gpu: (D, H, W) GPU float tensor (raw, unnormalized).
            centers: list of [cx, cy, cz].
            output_size: int, spatial size of each crop.

        Returns:
            (N, 1, output_size, output_size, output_size) GPU float tensor.
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

            # Boundary padding amounts
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
                # F.pad format: (W_left, W_right, H_left, H_right, D_left, D_right)
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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def new_init(self, img, center):
        """Initialize template features from the first frame.

        Crops two patches at different scales (reg_img_size for AE branch,
        dis_img_size for MoCo branch), normalizes each, and extracts
        template features.

        Args:
            img: 3D numpy array (D, H, W) -- the full volume.
            center: [cx, cy, cz] landmark position.
        """
        self.position = center

        # Transfer raw volume to GPU as float32, no normalization
        volume_gpu = self._volume_to_gpu(img)

        # Crop template patches
        z_reg = self._crop_gpu_batch(volume_gpu, [center], self.reg_img_size)  # (1,1,D,H,W)
        z_reg = self._normalize_patches(z_reg)

        # Extract template features (no grad)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            self.z1 = self.model_autoencoder.features(z_reg, False)  # spatial features

        # Only compute MoCo branch if needed
        if self.lambda_dis != 0 or self.use_adaptive_fusion or self.top_k_fusion > 0:
            z_dis = self._crop_gpu_batch(volume_gpu, [center], self.dis_img_size)
            z_dis = self._normalize_patches(z_dis)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                self.z2 = self.model_encoder(z_dis)
            self.z2 = self.z2.float()

        # Ensure template features are float32 for similarity computation
        self.z1 = self.z1.float()

    def init(self, img, center):
        """Initialize template features with single-size crop for both branches.

        Uses self.reg_img_size for both crops (API compatibility with MultiBranchTracker_18).

        Args:
            img: 3D numpy array (D, H, W) -- the full volume.
            center: [cx, cy, cz] landmark position.
        """
        self.position = center
        volume_gpu = self._volume_to_gpu(img)

        z = self._crop_gpu_batch(volume_gpu, [center], self.reg_img_size)
        z = self._normalize_patches(z)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            self.z1 = self.model_autoencoder.features(z, False)
            if self.lambda_dis != 0 or self.use_adaptive_fusion:
                self.z2 = self.model_encoder(z)
                self.z2 = self.z2.float()
        self.z1 = self.z1.float()

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def new_track(self, volume, step=1):
        """Track one frame using 26-neighbor greedy search with batch inference.

        All 27 candidates (center + 26 neighbors) are cropped and inferred
        in a single batch per greedy iteration. Cache prevents re-evaluating
        positions already scored in earlier iterations.

        When lambda_dis=0, only the AE branch is computed (skips MoCo crop
        and forward pass for ~25ms savings per frame).

        Args:
            volume: 3D numpy array (D, H, W) -- the current frame.
            step: search step size (voxels).

        Returns:
            (position, similarity) tuple.
        """
        volume_gpu = self._volume_to_gpu(volume)
        use_moco = (self.lambda_dis != 0) or self.use_adaptive_fusion or (self.top_k_fusion > 0)
        use_topk = self.top_k_fusion > 0
        p_0 = self.position
        self.cache.clear()

        max_iter = 10  # prevent infinite loops from oscillating scores
        for _iter in range(max_iter):
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)

            # Separate cached vs uncached candidates
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
                # AE branch: always crop + normalize + forward
                batch_reg = self._crop_gpu_batch(volume_gpu, uncached_candidates, self.reg_img_size)
                batch_reg = self._normalize_patches(batch_reg)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    x1_batch = self.model_autoencoder.features(batch_reg, False)

                ae_sim = self.calculate_cosine_similarity_batch(self.z1, x1_batch.float())

                if use_topk:
                    # Top-K fusion: MoCo only on top-K AE candidates
                    k = min(self.top_k_fusion, len(ae_sim))
                    topk_idx = ae_sim.topk(k).indices
                    topk_candidates = [uncached_candidates[i] for i in topk_idx]

                    batch_dis = self._crop_gpu_batch(volume_gpu, topk_candidates, self.dis_img_size)
                    batch_dis = self._normalize_patches(batch_dis)
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        x2_k = self.model_encoder(batch_dis)
                    mo_sim_k = self.calculate_ncc_batch(self.z2, x2_k.float())

                    # Z-normalize both branches for top-K
                    ae_sim_k = ae_sim[topk_idx]
                    ae_z = (ae_sim_k - ae_sim_k.mean()) / (ae_sim_k.std() + 1e-8)
                    mo_z = (mo_sim_k - mo_sim_k.mean()) / (mo_sim_k.std() + 1e-8)
                    fused_k = self.top_k_weight_ae * ae_z + (1 - self.top_k_weight_ae) * mo_z

                    # Use fused score for top-K, AE score for rest
                    batch_similarities = ae_sim.clone()
                    for j, orig_i in enumerate(topk_idx):
                        batch_similarities[orig_i] = fused_k[j]

                elif use_moco:
                    # Standard dual-branch fusion
                    batch_dis = self._crop_gpu_batch(volume_gpu, uncached_candidates, self.dis_img_size)
                    batch_dis = self._normalize_patches(batch_dis)
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        x2_batch = self.model_encoder(batch_dis)

                    if self.use_adaptive_fusion:
                        batch_similarities = self.calculate_similarity_batch_adaptive(
                            x1_batch.float(), x2_batch.float()
                        )
                    else:
                        batch_similarities = self.calculate_similarity_batch(
                            x1_batch.float(), x2_batch.float(),
                            lambda_1=self.lambda_reg,
                            lambda_2=self.lambda_dis
                        )
                else:
                    # AE-only mode
                    batch_similarities = ae_sim

                # Update cache and collect results
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

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def calculate_similarity_batch(self, x1_batch, x2_batch, lambda_1=1.0, lambda_2=1.0):
        """Weighted combination of AE cosine similarity and MoCo NCC.

        Args:
            x1_batch: AE features for candidates, (N, ...).
            x2_batch: MoCo encoder features for candidates, (N, C, D, H, W).
            lambda_1: weight for AE cosine similarity.
            lambda_2: weight for MoCo NCC.

        Returns:
            (N,) tensor of combined similarity scores.
        """
        ae_sim = self.calculate_cosine_similarity_batch(self.z1, x1_batch)
        e_sim = self.calculate_ncc_batch(self.z2, x2_batch)
        return lambda_1 * ae_sim + lambda_2 * e_sim

    def calculate_similarity_batch_adaptive(self, x1_batch, x2_batch):
        """Adaptive fusion weights based on branch discriminability.

        Uses coefficient of variation (CV = std / |mean|) to measure how well
        each branch distinguishes among candidates. Higher CV means the branch
        gives more varied scores (better discrimination), so it gets higher weight.

        Softmax temperature controls sharpness:
          temperature < 1.0 → sharper (more winner-take-all)
          temperature > 1.0 → smoother (closer to equal weights)

        Args:
            x1_batch: AE features for candidates, (N, ...).
            x2_batch: MoCo encoder features for candidates, (N, C, D, H, W).

        Returns:
            (N,) tensor of combined similarity scores.
        """
        ae_sim = self.calculate_cosine_similarity_batch(self.z1, x1_batch)  # (N,)
        e_sim = self.calculate_ncc_batch(self.z2, x2_batch)  # (N,)

        eps = 1e-8

        # Coefficient of variation for each branch
        ae_cv = ae_sim.std() / (ae_sim.mean().abs() + eps)
        e_cv = e_sim.std() / (e_sim.mean().abs() + eps)

        # Softmax weighting with temperature
        logits = torch.stack([ae_cv, e_cv]) / self.fusion_temperature
        weights = F.softmax(logits, dim=0)
        w_ae, w_dis = weights[0], weights[1]

        return w_ae * ae_sim + w_dis * e_sim

    def calculate_cosine_similarity_batch(self, z, x_batch):
        """Batch cosine similarity between template z and N candidates.

        Args:
            z: template feature, any shape (e.g. (1, 256) or (1, C, D, H, W)).
            x_batch: candidate features, (N, ...).

        Returns:
            (N,) tensor of cosine similarities.
        """
        N = x_batch.shape[0]
        if z.dim() == 1:
            z = z.unsqueeze(0)    # (1, C)
        elif z.dim() >= 2:
            z = z.view(1, -1)     # (1, C_total)

        if x_batch.dim() >= 2:
            x_batch = x_batch.view(N, -1)  # (N, C_total)

        z_expanded = z.expand(N, -1)
        cos_sim = F.cosine_similarity(z_expanded, x_batch, dim=1)
        return cos_sim

    def calculate_ncc_batch(self, z, x_batch, eps=1e-8):
        """Per-channel batch NCC between template z and N candidates.

        Computes NCC independently for each channel, then averages across
        channels. When use_weighted_ncc is True, channels are weighted by
        their discriminability (std of per-channel NCC across candidates).

        Matches MoCoStudentScorer.calculate_ncc_single behavior.

        Args:
            z: template feature, (B, C, D, H, W).
            x_batch: candidate features, (N, C, D, H, W).
            eps: numerical stability constant.

        Returns:
            (N,) tensor of NCC scores.
        """
        if self.use_weighted_ncc:
            return self.calculate_ncc_batch_weighted(z, x_batch, eps)

        N = x_batch.shape[0]
        C = z.shape[1]

        # Expand z to match batch
        if z.shape[0] == 1:
            z_expanded = z.expand(N, -1, -1, -1, -1)  # (N, C, D, H, W)
        else:
            z_expanded = z.repeat_interleave(N // z.shape[0], dim=0)

        # Flatten spatial dims: (N, C, L)
        z_flat = z_expanded.view(N, C, -1)
        x_flat = x_batch.view(N, C, -1)

        # Per-channel mean
        z_mean = z_flat.mean(dim=2, keepdim=True)  # (N, C, 1)
        x_mean = x_flat.mean(dim=2, keepdim=True)

        z_centered = z_flat - z_mean
        x_centered = x_flat - x_mean

        # Per-channel covariance and variance
        cov = (z_centered * x_centered).sum(dim=2)  # (N, C)
        z_var = (z_centered ** 2).sum(dim=2)  # (N, C)
        x_var = (x_centered ** 2).sum(dim=2)  # (N, C)

        # Per-channel NCC
        ncc_per_channel = cov / torch.sqrt(z_var * x_var + eps)  # (N, C)
        ncc = ncc_per_channel.mean(dim=1)  # (N,)

        return ncc

    def calculate_ncc_batch_weighted(self, z, x_batch, eps=1e-8):
        """Per-channel NCC with variance-based channel weighting.

        Channels whose NCC scores vary more across candidates are more
        discriminative for distinguishing among them. Weight = std(ncc_per_channel)
        across candidates, normalized to sum to 1.

        Args:
            z: template feature, (B, C, D, H, W).
            x_batch: candidate features, (N, C, D, H, W).
            eps: numerical stability constant.

        Returns:
            (N,) tensor of weighted NCC scores.
        """
        N = x_batch.shape[0]
        C = z.shape[1]

        if z.shape[0] == 1:
            z_expanded = z.expand(N, -1, -1, -1, -1)
        else:
            z_expanded = z.repeat_interleave(N // z.shape[0], dim=0)

        z_flat = z_expanded.view(N, C, -1)
        x_flat = x_batch.view(N, C, -1)

        z_mean = z_flat.mean(dim=2, keepdim=True)
        x_mean = x_flat.mean(dim=2, keepdim=True)

        z_centered = z_flat - z_mean
        x_centered = x_flat - x_mean

        cov = (z_centered * x_centered).sum(dim=2)
        z_var = (z_centered ** 2).sum(dim=2)
        x_var = (x_centered ** 2).sum(dim=2)

        ncc_per_channel = cov / torch.sqrt(z_var * x_var + eps)  # (N, C)

        # Channel weighting: weight by std across candidates
        channel_std = ncc_per_channel.std(dim=0)  # (C,)
        channel_std = channel_std / (channel_std.sum() + eps)  # normalize to sum=1

        ncc = (ncc_per_channel * channel_std.unsqueeze(0)).sum(dim=1)  # (N,)
        return ncc

    def _generate_6_neighbors(self, pos, step):
        cx, cy, cz = pos
        candidates = [[cx, cy, cz]]
        candidates.extend([
            [cx + step, cy, cz],    # +x
            [cx - step, cy, cz],    # -x
            [cx, cy + step, cz],    # +y
            [cx, cy - step, cz],    # -y
            [cx, cy, cz + step],    # +z
            [cx, cy, cz - step],    # -z
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

    def find_best_position(self, p_0, p_max):
        """Check convergence: True if position hasn't changed."""
        return (p_0[0] == p_max[0] and
                p_0[1] == p_max[1] and
                p_0[2] == p_max[2])
