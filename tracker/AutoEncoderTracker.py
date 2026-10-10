import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils.generate_channels import generate_channels, get_generate_channels_function
class AutoEncoderTracker(nn.Module):
    def __init__(self, model, device, method, img_size, channels_24_direction='z', normalize=True, mlp=True):
        super(AutoEncoderTracker, self).__init__()
        self.model = model

        self.device = device
        self.img_size = img_size
        self.in_channels = model.get_in_channels()
        self._set_model(model)
        self.generate_channels_function = get_generate_channels_function(method, self.in_channels, channels_24_direction)
        self.channels_24_direction = channels_24_direction

        self.normalize = normalize
        self.mlp = mlp

    def _set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()



    def init(self, volume, center):
        """

        :param volume:
        :param center:
        :return:
        """
        self.position = center
        z = self.generate_channels_function(volume, center, self.img_size)
        z = torch.from_numpy(z).float().to(self.device)
        # for i in range(9):
        #     print(z[i,:,:].max())
        #     plt.imshow(z[i, :, :].cpu(), cmap='gray')
        #     plt.show()
        z = z.unsqueeze(0)
        if self.normalize:
            z = z / 255.0
        #print(z.shape)
        self.z_r = self.model.features(z, self.mlp)


    @torch.no_grad()
    def track(self, volume, step=1):
        """

        :param volume:
        :return:
        """
        p_0 = self.position
        p_min = [-1, -1, -1]
        while (True):
            distance = float('inf')
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)
            # 6-neighborhood
            # candidates = [
            #     [cx, cy, cz],
            #     [cx + 1, cy, cz],
            #     [cx - 1, cy, cz],
            #     [cx, cy + 1, cz],
            #     [cx, cy - 1, cz],
            #     [cx, cy, cz + 1],
            #     [cx, cy, cz - 1],
            # ]
            for candidate in candidates:
                #print("candidate: {}".format(candidate))
                image_search = self.generate_channels_function(volume, candidate, self.img_size)
                image_search = torch.from_numpy(image_search).float().to(self.device)
                image_search = image_search.unsqueeze(0)
                if self.normalize:
                    image_search = image_search / 255.0
                z_k = self.model.features(image_search, self.mlp)
                #cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                #cur_distance = self.calculate_distance(self.z_r, z_k, p_0, candidate)
                #cur_distance = self._correlation_distance(self.z_r, z_k)
                cur_distance = self.calculate_ncc(self.z_r, z_k)
                if cur_distance < distance:
                    #print("candidate: {}, min distance: {}".format(candidate, cur_distance))
                    distance = cur_distance
                    p_min = candidate
            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        self.position = p_min
        return self.position

    # 26-neighborhood
    def _generate_search_region(self, pos, step):
        cx, cy, cz = pos
        candidates = []
        candidates.append([cx, cy, cz])
        deltas = [0, step, -step]
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_x = cx + dx
                    neighbor_y = cy + dy
                    neighbor_z = cz + dz
                    candidates.append([neighbor_x, neighbor_y, neighbor_z])
        return candidates

    def _correlation_distance(self, z_r, z_k):
        z_r = z_r.view(1, 256, 2, 2, 2)
        z_k = z_k.view(1, 256, 2, 2, 2)
        weight = z_r.clone()
        out_global = F.conv3d(z_k, weight)  # output shape = (B,1,1,1,1)
        similarity_score = out_global.view(1)
        return 1-similarity_score

    def calculate_cosine_distance(self, z_r, z_k):
        cos_sim = F.cosine_similarity(z_r, z_k, dim=1)
        return 1 - cos_sim

    def calculate_spatial_distance(self, p_0, p_r):
        distance = np.sqrt((p_0[0] - p_r[0]) ** 2 + (p_0[1] - p_r[1]) ** 2 + (p_0[2] - p_r[2]) ** 2)
        return distance

    def calculate_distance(self, z_r, z_k, p_0, p_r, lambda_spatial=0.000):
        appearance_distance = self.calculate_cosine_distance(z_r, z_k)
        spatial_distance = self.calculate_spatial_distance(self.position, p_r)
        distance = appearance_distance + lambda_spatial * spatial_distance
        return distance

    def find_best_position(self, p_0, p_min):
        if p_0[0] == p_min[0] and p_0[1] == p_min[1] and p_0[2] == p_min[2]:
            return True
        return False

    def is_need_search(self, p_0, p_min):
        if p_0[0] == p_min[0] and p_0[1] == p_min[1] and p_0[2] == p_min[2]:
            return True
        return False

    def calculate_ncc(self, z, x, eps=1e-8):
        # Automatically add batch and channel dimensions if missing
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            z = z.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(0)  # (1, C, D, H, W)
            z = z.unsqueeze(0)

        B, C, D, H, W = z.shape
        z_flat = z.view(B, C, -1)
        x_flat = x.view(B, C, -1)

        # Compute the mean (along the spatial dimensions)
        x_mean = x_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
        z_mean = z_flat.mean(dim=2, keepdim=True)

        # Subtract the mean
        x_centered = x_flat - x_mean
        z_centered = z_flat - z_mean

        # Numerator: covariance term
        cov = (x_centered * z_centered).sum(dim=2)  # (B, C)

        # Denominator: product of the respective standard deviations
        x_var = (x_centered ** 2).sum(dim=2)
        z_var = (z_centered ** 2).sum(dim=2)
        std_prod = torch.sqrt(x_var * z_var + eps)  # (B, C)

        # NCC per channel
        ncc_per_channel = cov / std_prod  # (B, C)

        # Optional: average over channels (or keep per-channel)
        ncc = ncc_per_channel.mean(dim=1)  # (B,)

        # If the original input has no batch dimension, return a scalar
        if B == 1:
            return -ncc[0].item()
        else:
            return -ncc