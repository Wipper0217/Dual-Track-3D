import numpy as np
import torch
import torch.nn as nn
from utils.generate_channels import generate_channels
from model.builder.AutoEncoder import AutoEncoder
from sklearn.metrics.pairwise import cosine_distances

class AutoEncoderTracker(nn.Module):
    def __init__(self, in_channels, channel_24_direction=None):
        super(AutoEncoderTracker, self).__init__()
        self.in_channels = in_channels
        self.model = AutoEncoder(self.in_channels)

        self.channel_24_direction = channel_24_direction

    def init(self, image, center):
        image_template = generate_channels(image, center, self.in_channels, self.channel_24_direction)
        self.z_r = self.model.features(image_template)
        self.position = center

    @torch.no_grad()
    def track(self, image):
        p_0 = self.position

        p_k = [-1, -1, -1]

        while (self.is_need_search(p_0, p_k)):
            distance = 3
            cx, cy, cz = p_0
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
                image_search = generate_channels(image, candidate, self.in_channels, self.channel_24_direction)
                z_k = self.model.features(image_search)
                cur_distance = self.calculate_cosine_distance(self.z_r, z_k)
                if cur_distance < distance:
                    distance = cur_distance
                    p_0 = candidate

        self.position = p_0
        return self.position




    def calculate_cosine_distance(self, z_r, z_k):
        z_r = np.array([z_r])
        z_k = np.array([z_k])
        distance = cosine_distances(z_r, z_k)[0, 0]
        return distance


    def get_search_candidates(self, center, search_radius=1):
        cx, cy, cz = center
        candidate_x = [cx + i for i in range(-search_radius, search_radius + 1)]
        candidate_y = [cy + i for i in range(-search_radius, search_radius + 1)]
        candidate_z = [cz + i for i in range(-search_radius, search_radius + 1)]
        return candidate_x, candidate_y, candidate_z

    def is_need_search(self, p_0, p_min):
        if p_0[0] == p_min[0] and p_0[1] == p_min[1] and p_0[2] == p_min[2]:
            return True
        return False


