import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.transforms import Compose, ScaleIntensityd

from contrastive_learning.dataset.crop_utils import crop_and_resize_numpy, crop_numpy
class MocoEncoderTracker(nn.Module):
    def __init__(self, model, device, img_size, normalize=True, lambda_spatial=0.0):
        super(MocoEncoderTracker, self).__init__()
        self.model = model

        self.device = device
        self.img_size = img_size
        self._set_model(model)
        self.generate_channels_function = crop_numpy

        self.normalize = normalize
        self.lambda_spatial = lambda_spatial

        self.size = [20, 20, 20]


    def _set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()

    def get_name(self):
        return "MocoEncoderTracker"

    def pre_process(self, image):
        data_dict = {
            "image": image
        }
        trasnform = Compose([
            # Normalize to [0, 1]
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])
        transform_img = trasnform(data_dict)
        return transform_img['image'].as_tensor().float()

    def old_init(self, img, center):
        """

        :param volume:
        :param center:
        :return:
        """
        self.position = center
        # z = crop_and_resize_numpy(
        #     img=img,
        #     center=center,
        #     size=self.size,
        #     output_size=self.img_size,
        #     context_amount=0.5
        # )
        z = crop_numpy(
            img=img,
            center=center,
            output_size=self.img_size
        )


        z = torch.from_numpy(z).float().to(self.device)
        z = z.unsqueeze(0).unsqueeze(0)
        if self.normalize:
            z = z / 255.0
        self.z_r = self.model(z).view(1, -1)
        self.init_cache()

    def init(self, img, center):
        """

        :param volume:
        :param center:
        :return:
        """
        self.position = center
        # z = crop_and_resize_numpy(
        #     img=img,
        #     center=center,
        #     size=self.size,
        #     output_size=self.img_size,
        #     context_amount=0.5
        # )
        z = crop_numpy(
            img=img,
            center=center,
            output_size=self.img_size
        )

        z = self.pre_process(z)
        z = z.to(self.device)
        z = z.unsqueeze(0).unsqueeze(0)

        self.z_r = self.model(z)#.view(1, -1)
        self.init_cache()

    def init_cache(self):
        self.cache = {}

    @torch.no_grad()
    def old_track(self, volume, step=1):
        """

        :param volume:
        :return:
        """
        p_0 = self.position
        p_min = [-1, -1, -1]
        while (True):
            distance = float('inf')
            cx, cy, cz = p_0
            #candidates = self._generate_search_region([cx, cy, cz], step)
            candidates = [
                [cx, cy, cz],
                [cx + 1, cy, cz],
                [cx - 1, cy, cz],
                [cx, cy + 1, cz],
                [cx, cy - 1, cz],
                [cx, cy, cz + 1],
                [cx, cy, cz - 1],
            ]
            for candidate in candidates:
                # print("candidate: {}".format(candidate))
                # image_search = crop_and_resize_numpy(
                #     img=volume,
                #     center=candidate,
                #     size=self.size,
                #     output_size=self.img_size,
                #     context_amount=0.5
                # )
                image_search = crop_numpy(
                    img=volume,
                    center=candidate,
                    output_size=self.img_size
                )
                image_search = torch.from_numpy(image_search).float().to(self.device)
                image_search = image_search.unsqueeze(0).unsqueeze(0)
                if self.normalize:
                    image_search = image_search / 255.0
                z_k = self.model(image_search).view(1, -1)
                # cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                cur_distance = self.calculate_distance(self.z_r, z_k, p_0, candidate, self.lambda_spatial)
                # print("candidate: {}, distance: {}".format(candidate, cur_distance))
                if cur_distance < distance:
                    # print("candidate: {}, min distance: {}".format(candidate, cur_distance))
                    distance = cur_distance
                    p_min = candidate
            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        self.position = p_min
        return self.position

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
                # image_search = crop_and_resize_numpy(
                #     img=volume,
                #     center=candidate,
                #     size=self.size,
                #     output_size=self.img_size,
                #     context_amount=0.5
                # )
                image_search = crop_numpy(
                    img=volume,
                    center=candidate,
                    output_size=self.img_size
                )
                image_search = self.pre_process(image_search)
                #image_search = torch.from_numpy(image_search).float().to(self.device)
                image_search = image_search.to(self.device)
                image_search = image_search.unsqueeze(0).unsqueeze(0)

                z_k = self.model(image_search)#.view(1, -1)
                #cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                #cur_distance = self.calculate_distance(self.z_r, z_k, p_0, candidate, self.lambda_spatial)
                cur_distance = self.calculate_ncc(self.z_r, z_k)
                #print("candidate: {}, distance: {}".format(candidate, cur_distance))
                if cur_distance < distance:
                    #print("candidate: {}, min distance: {}".format(candidate, cur_distance))
                    distance = cur_distance
                    p_min = candidate
            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        self.position = p_min

        return self.position, distance

    def greedy_search(self, volume, pos, step):
        """
        Greedy search
        :param volume:
        :return:
        """
        p_0 = pos
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
                # print("candidate: {}".format(candidate))
                # image_search = crop_and_resize_numpy(
                #     img=volume,
                #     center=candidate,
                #     size=self.size,
                #     output_size=self.img_size,
                #     context_amount=0.5
                # )
                image_search = crop_numpy(
                    img=volume,
                    center=candidate,
                    output_size=self.img_size
                )
                image_search = self.pre_process(image_search)
                image_search = image_search.to(self.device)
                image_search = image_search.unsqueeze(0).unsqueeze(0)
                # todo: normalization
                z_k = self.model(image_search).view(1, -1)
                # cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                cur_distance = self.calculate_distance(self.z_r, z_k, p_0, candidate, lambda_spatial=0.0)
                key = self._generate_key(candidate)
                self.cache[key] = cur_distance
                # print("candidate: {}, distance: {}".format(candidate, cur_distance))
                if cur_distance < distance:
                    # print("candidate: {}, min distance: {}".format(candidate, cur_distance))
                    distance = cur_distance
                    p_min = candidate
            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        return p_min, distance
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


    def calculate_cosine_distance(self, z_r, z_k):
        cos_sim = F.cosine_similarity(z_r, z_k, dim=1)
        return 1 - cos_sim

    def calculate_spatial_distance(self, p_0, p_r):
        distance = np.sqrt((p_0[0] - p_r[0]) ** 2 + (p_0[1] - p_r[1]) ** 2 + (p_0[2] - p_r[2]) ** 2)
        return distance


    def calculate_distance(self, z_r, z_k, p_0, p_r, lambda_spatial=0.001):
        appearance_distance = self.calculate_cosine_distance(z_r, z_k)
        #spatial_distance = self.calculate_spatial_distance(self.position, p_r)
        spatial_distance = self.calculate_spatial_distance(p_0, p_r)
        distance = appearance_distance + lambda_spatial * spatial_distance
        return distance

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
        """
        Coarse-grained search
        :param volume:
        :param search_radius:
        :param next_search_amount:
        :return:
        """
        last_pos = self.position
        next_search_centers = []
        candidates_pos = self._generate_search_region(last_pos, search_radius)
        for candidate in candidates_pos:
            # todo: extract into a method
            image_search = crop_numpy(
                img=volume,
                center=candidate,
                output_size=self.img_size
            )

            image_search = self.pre_process(image_search)
            image_search = image_search.to(self.device)
            image_search = image_search.unsqueeze(0).unsqueeze(0)

            z_k = self.model(image_search).view(1, -1)
            cur_distance = self.calculate_distance(self.z_r, z_k, last_pos, candidate)
            key = self._generate_key(candidate)
            self.cache[key] = cur_distance
            next_search_centers.append([candidate, cur_distance])

        next_search_centers = sorted(next_search_centers, key=lambda x: x[1])
        return next_search_centers[:next_search_amount]

    def _generate_key(self, pos):
        cx, cy, cz = map(round, pos)
        key = 'pos_' + str(cx) + '_' + str(cy) + '_' + str(cz)
        return key

    def fine_search(self, volume, next_search_centers):
        """
        Fine-grained search
        :param volume:
        :param next_search_centers:
        :return:
        """
        candidate_predict = []
        for candidate, distance in next_search_centers:
            p_min, distance = self.greedy_search(volume, candidate, 1)
            candidate_predict.append([p_min, distance])

        candidate_predict = sorted(candidate_predict, key=lambda x: x[1])
        return candidate_predict[0][0]
