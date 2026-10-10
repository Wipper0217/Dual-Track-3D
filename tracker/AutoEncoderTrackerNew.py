import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from contrastive_learning.dataset.crop_utils import crop_and_resize_numpy
class AutoEncoderTrackerNew(nn.Module):
    def __init__(self, model, device, img_size, normalize=True, mlp=True, size=[18, 18, 18]):
        super(AutoEncoderTrackerNew, self).__init__()
        self.model = model

        self.device = device
        self.img_size = img_size
        self.in_channels = model.get_in_channels()
        self._set_model(model)
        self.generate_channels_function = crop_and_resize_numpy

        self.normalize = normalize
        self.mlp = mlp
        self.size = size

    def _set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()



    def init(self, img, center):
        """

        :param volume:
        :param center:
        :return:
        """
        self.position = center
        z = crop_and_resize_numpy(
            img=img,
            center=center,
            size=self.size,
            output_size=self.img_size,
            context_amount=0.5
        )
        z = torch.from_numpy(z).float().to(self.device)
        z = z.unsqueeze(0).unsqueeze(0)
        if self.normalize:
            z = z / 255.0
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
                image_search = crop_and_resize_numpy(
                    img=volume,
                    center=candidate,
                    size=self.size,
                    output_size=self.img_size,
                    context_amount=0.5
                )
                image_search = torch.from_numpy(image_search).float().to(self.device)
                image_search = image_search.unsqueeze(0).unsqueeze(0)
                if self.normalize:
                    image_search = image_search / 255.0
                z_k = self.model.features(image_search, self.mlp)
                #cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                cur_distance = self.calculate_distance(self.z_r, z_k, p_0, candidate)
                #print("candidate: {}, distance: {}".format(candidate, cur_distance))
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

    def calculate_cosine_distance(self, z_r, z_k):
        cos_sim = F.cosine_similarity(z_r, z_k, dim=1)
        return 1 - cos_sim

    def calculate_spatial_distance(self, p_0, p_r):
        distance = np.sqrt((p_0[0] - p_r[0]) ** 2 + (p_0[1] - p_r[1]) ** 2 + (p_0[2] - p_r[2]) ** 2)
        return distance

    def calculate_distance(self, z_r, z_k, p_0, p_r, lambda_spatial=0.001):
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