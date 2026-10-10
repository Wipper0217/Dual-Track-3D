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
        trasnform = Compose([
            # Normalize to [0, 1]
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ])
        transform_img = trasnform(data_dict)
        return transform_img['image'].as_tensor().float()

    def new_init(self, img, center):
        """

        :param volume:
        :param center:
        :return:
        """
        self.position = center
        z_reg = crop_numpy(
            img=img,
            center=center,
            output_size=self.reg_img_size
        )

        z_dis = crop_numpy(
            img=img,
            center=center,
            output_size=self.dis_img_size
        )

        z_reg = self.pre_process(z_reg)
        z_reg = z_reg.to(self.device)
        z_reg = z_reg.unsqueeze(0).unsqueeze(0)

        z_dis = self.pre_process(z_dis)
        z_dis = z_dis.to(self.device)
        z_dis = z_dis.unsqueeze(0).unsqueeze(0)

        self.z1 = self.model_autoencoder.features(z_reg, True)
        self.z2 = self.model_encoder(z_dis)

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

        self.z1 = self.model_autoencoder.features(z, True)
        self.z2 = self.model_encoder(z)

    def _init_search_cache(self):
        self.cache = {}

    @torch.no_grad()
    def track(self, volume, step=1):
        """

        :param volume:
        :return:
        """
        # p_0 = self.position
        # p_min = [-1, -1, -1]
        # while (True):
        #     similarity = float('-inf')
        #     cx, cy, cz = p_0
        #     candidates = self._generate_search_region([cx, cy, cz], step)
        #
        #     # For ablation experiments
        #     # candidates = [
        #     #     [cx, cy, cz],
        #     #     [cx + 1, cy, cz],
        #     #     [cx - 1, cy, cz],
        #     #     [cx, cy + 1, cz],
        #     #     [cx, cy - 1, cz],
        #     #     [cx, cy, cz + 1],
        #     #     [cx, cy, cz - 1],
        #     # ]
        #
        #     for candidate in candidates:
        #         # print("candidate: {}".format(candidate))
        #         # image_search = crop_and_resize_numpy(
        #         #     img=volume,
        #         #     center=candidate,
        #         #     size=self.size,
        #         #     output_size=self.img_size,
        #         #     context_amount=0.5
        #         # )
        #         image_search = crop_numpy(
        #             img=volume,
        #             center=candidate,
        #             output_size=self.img_size
        #         )
        #         image_search = self.pre_process(image_search)
        #         image_search = image_search.to(self.device)
        #         image_search = image_search.unsqueeze(0).unsqueeze(0)
        #
        #         x1 = self.model_autoencoder.features(image_search, True)
        #         x2 = self.model_encoder(image_search)
        #         cur_similarity = self.calculate_similarity(x1, x2, lambda_1=1.0,  lambda_2=0.0)               # print("candidate: {}, distance: {}".format(candidate, cur_distance))
        #         if cur_similarity > similarity:
        #             similarity = cur_similarity
        #             p_min = candidate
        #     if self.find_best_position(p_0, p_min):
        #         break
        #     p_0 = p_min
        # self.position = p_min
        #
        # return self.position, similarity
        return self.status_cache_search(volume, step)

    def new_track(self, volume, step=1):
        p_0 = self.position
        p_max = [-1, -1, -1]
        # Clear the cache
        self.cache.clear()
        while (True):
            similarity = float('-inf')
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
                key = self.generate_cache_key(candidate)
                cur_similarity = self.cache.get(key, None)
                if cur_similarity is None:
                    image_search_reg = crop_numpy(
                        img=volume,
                        center=candidate,
                        output_size=self.reg_img_size
                    )
                    image_search_dis = crop_numpy(
                        img=volume,
                        center=candidate,
                        output_size=self.dis_img_size
                    )

                    image_search_reg = self.pre_process(image_search_reg)
                    image_search_reg = image_search_reg.to(self.device)
                    image_search_reg = image_search_reg.unsqueeze(0).unsqueeze(0)

                    x1 = self.model_autoencoder.features(image_search_reg, True)

                    image_search_dis = self.pre_process(image_search_dis)
                    image_search_dis = image_search_dis.to(self.device)
                    image_search_dis = image_search_dis.unsqueeze(0).unsqueeze(0)
                    x2 = self.model_encoder(image_search_dis)
                    cur_similarity = self.calculate_similarity(x1, x2,
                                                               lambda_1=self.lambda_reg,
                                                               lambda_2=self.lambda_dis)
                    self.cache[key] = cur_similarity
                if cur_similarity > similarity:
                    similarity = cur_similarity
                    p_max = candidate
            if self.find_best_position(p_0, p_max):
                break
            p_0 = p_max
        self.position = p_max

        return self.position, similarity

    def status_cache_search(self, volume, step=1):
        p_0 = self.position
        p_max = [-1, -1, -1]
        # Clear the cache
        self.cache.clear()
        while (True):
            similarity = float('-inf')
            cx, cy, cz = p_0
            candidates = self._generate_search_region([cx, cy, cz], step)
            # For ablation experiments
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
                key = self.generate_cache_key(candidate)
                cur_similarity = self.cache.get(key, None)
                if cur_similarity is None:
                    image_search = crop_numpy(
                        img=volume,
                        center=candidate,
                        output_size=self.img_size
                    )
                    image_search = self.pre_process(image_search)
                    image_search = image_search.to(self.device)
                    image_search = image_search.unsqueeze(0).unsqueeze(0)

                    x1 = self.model_autoencoder.features(image_search, True)
                    x2 = self.model_encoder(image_search)
                    cur_similarity = self.calculate_similarity(x1, x2, lambda_1=0.2,
                                                               lambda_2=0.8)
                    self.cache[key] = cur_similarity
                if cur_similarity > similarity:
                    similarity = cur_similarity
                    p_max = candidate
            if self.find_best_position(p_0, p_max):
                break
            p_0 = p_max
        self.position = p_max

        return self.position, similarity

    def calculate_similarity(self, x1, x2, lambda_1=1.0, lambda_2=1.0):
        ae_sim = self.calculate_cosine_similarity(self.z1, x1)
        e_sim = self.calculate_ncc(self.z2, x2)
        #print("ae sim: {}, moco sim: {}".format(ae_sim, e_sim))
        #e_sim = self.calculate_cosine_similarity(self.z2, x2)
        return lambda_1 * ae_sim + lambda_2 * e_sim

    def calculate_cosine_similarity(self, z, x):
        z = z.view(1, -1)
        x = x.view(1, -1)
        cos_sim = F.cosine_similarity(z, x, dim=1)
        return cos_sim


    def calculate_relative_similarity(self, z, x):
        """
        Compute the relative similarity relateive_sim(x) = sim(x) - mean(sim(neighborhood(x))
        :param z:
        :param x:
        :return:
        """
        sim_x = self.calculate_cosine_similarity(z, x)


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
            return ncc[0].item()
        else:
            return ncc

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

            x = self.model_encoder(image_search)
            cur_sim = self.calculate_cosine_similarity(self.z2, x)
            key = self._generate_key(candidate)
            #self.cache[key] = cur_sim
            next_search_centers.append([candidate, cur_sim])

        next_search_centers = sorted(next_search_centers, key=lambda x: x[1], reverse=True)
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
            p_min, similarity = self._greedy_search(volume, candidate, 1)
            candidate_predict.append([p_min, similarity])

        candidate_predict = sorted(candidate_predict, key=lambda x: x[1], reverse=True)
        return candidate_predict[0][0]

    def _greedy_search(self, volume, pos, step):
        """
        Greedy search
        :param volume:
        :return:
        """
        p_0 = pos
        p_min = [-1, -1, -1]
        while (True):
            similarity = float('-inf')
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
                x1 = self.model_autoencoder.features(image_search, True)
                x2 = self.model_encoder(image_search)
                cur_similarity = self.calculate_similarity(x1, x2)
                # print("candidate: {}, distance: {}".format(candidate, cur_distance))
                if cur_similarity > similarity:
                    similarity = cur_similarity
                    p_min = candidate
                key = self._generate_key(candidate)
                #self.cache[key] = cur_similarity
                # print("candidate: {}, distance: {}".format(candidate, cur_distance))

            if self.find_best_position(p_0, p_min):
                break
            p_0 = p_min
        return p_min, similarity