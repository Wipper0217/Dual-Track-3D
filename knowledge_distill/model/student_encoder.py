"""
Student network: 3D ResNet-18 processing 64³ input
Distilled from the MoCo ResNet-50 teacher network

Design notes:
- Input size 64³, consistent with the teacher network
- Outputs 512-channel 3D feature maps [B, 512, D, H, W]
- A 1×1 convolution projection head maps the 512-dimensional features to a 2048-dimensional space, used for feature-level distillation
- At inference, only the 512-dimensional original features are used to compute NCC; the projection head is not needed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3x3 convolution"""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3,
        dilation=dilation, stride=stride, padding=dilation, bias=False
    )


class BasicBlock(nn.Module):
    """ResNet BasicBlock for 3D"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StudentMoCoEncoder(nn.Module):
    """
    Student network: 3D ResNet-18
    Input: [B, 1, 64, 64, 64]
    Output: [B, 512, D, H, W] spatial feature maps

    The network structure keeps the same spatial resolution design as the teacher ResNet-50:
    - layer3/layer4 use dilated convolution to preserve spatial resolution
    - The output feature map resolution is consistent with the teacher network (8x8x8)
    """
    def __init__(self, in_channels=1):
        super(StudentMoCoEncoder, self).__init__()
        self.name = "StudentMoCoEncoder"
        self.inplanes = 64

        # Stem: conv1 + bn1 + relu + maxpool
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 stages: BasicBlock × [2, 2, 2, 2] (ResNet-18)
        # layer3/layer4 use dilated conv to preserve spatial resolution, consistent with the teacher network
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=1, dilation=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=1, dilation=4)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: [B, 1, 64, 64, 64]
        Returns:
            [B, 512, 8, 8, 8] spatial feature maps

        Computation flow (input 64³):
            conv1: stride=2, 64→32
            maxpool: stride=2, 32→16
            layer1: stride=1, 16→16
            layer2: stride=2, 16→8
            layer3: stride=1, dilation=2, 8→8
            layer4: stride=1, dilation=4, 8→8
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MoCoStudentScorer(nn.Module):
    """
    Complete student scorer: Encoder + feature projection head + NCC

    Composition:
    1. StudentMoCoEncoder: outputs 512-dimensional spatial feature maps
    2. feature_projection: 1×1 Conv 512→2048, used for feature-level distillation (only used during training)
    3. NCC computation: used for response-level distillation and tracking inference

    At inference:
    - Only the 512-dimensional features output by the encoder are used to compute NCC
    - The projection head is not used, making it faster
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = StudentMoCoEncoder(in_channels)

        # 1×1 convolution projection head: 512 → 2048, preserving spatial resolution
        # Used only during distillation training, to align with the teacher's 2048-dimensional features
        self.feature_projection = nn.Sequential(
            nn.Conv3d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, template, candidates, return_projection=False):
        """
        Forward pass: extract template and candidate features, compute NCC

        Args:
            template: [B, 1, D, H, W] template image
            candidates: [B, 1, D, H, W] (single candidate) or [B*N, 1, D, H, W] (multiple candidates)
            return_projection: whether to return projected features (only used during distillation training)

        Returns:
            ncc_scores: [B] or [B*N] NCC similarity scores
            z_feat: [B, 512, D', H', W'] template original features
            x_feat: [B, 512, D', H', W'] or [B*N, 512, D', H', W'] candidate original features
            When return_projection=True, additionally returns:
            z_proj: [B, 2048, D', H', W'] template projected features
            x_proj: [B*N, 2048, D', H', W'] candidate projected features
        """
        z_feat = self.encoder(template)

        if candidates.dim() == 5 and candidates.shape[0] == z_feat.shape[0]:
            # Single candidate case
            x_feat = self.encoder(candidates)
            ncc_scores = self.calculate_ncc_single(z_feat, x_feat)
        else:
            # Multiple candidate case
            x_feat = self.encoder(candidates)
            ncc_scores = self.calculate_ncc_batch(z_feat, x_feat)

        if return_projection:
            z_proj = self.feature_projection(z_feat)
            x_proj = self.feature_projection(x_feat)
            return ncc_scores, z_feat, x_feat, z_proj, x_proj

        return ncc_scores, z_feat, x_feat

    @staticmethod
    def calculate_ncc_single(z, x, eps=1e-8):
        """
        Single-candidate NCC computation
        Args:
            z: [B, C, D, H, W] template features
            x: [B, C, D, H, W] candidate features
        Returns:
            ncc: [B] NCC scores
        """
        B, C, D, H, W = z.shape

        z_flat = z.view(B, C, -1)  # [B, C, L]
        x_flat = x.view(B, C, -1)  # [B, C, L]

        z_mean = z_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
        x_mean = x_flat.mean(dim=2, keepdim=True)

        z_centered = z_flat - z_mean
        x_centered = x_flat - x_mean

        cov = (z_centered * x_centered).sum(dim=2)  # [B, C]

        z_std = torch.sqrt((z_centered ** 2).sum(dim=2) + eps)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=2) + eps)

        ncc_per_channel = cov / (z_std * x_std + eps)  # [B, C]
        ncc = ncc_per_channel.mean(dim=1)  # [B]

        return ncc

    @staticmethod
    def calculate_ncc_batch(z, x, eps=1e-8):
        """
        Batch NCC computation (multiple candidates)
        Args:
            z: [B, C, D, H, W] template features
            x: [N_total, C, D, H, W] candidate features, N_total = B * N
        Returns:
            ncc: [N_total] NCC scores
        """
        B = z.shape[0]
        N_total = x.shape[0]
        N = N_total // B
        C = z.shape[1]

        # Expand the template to each candidate
        z = z.repeat_interleave(N, dim=0)  # [B*N, C, D, H, W]

        z_flat = z.view(B * N, C, -1)  # [B*N, C, L]
        x_flat = x.view(B * N, C, -1)  # [B*N, C, L]

        z_mean = z_flat.mean(dim=2, keepdim=True)
        x_mean = x_flat.mean(dim=2, keepdim=True)

        z_centered = z_flat - z_mean
        x_centered = x_flat - x_mean

        cov = (z_centered * x_centered).sum(dim=2)  # [B*N, C]

        z_std = torch.sqrt((z_centered ** 2).sum(dim=2) + eps)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=2) + eps)

        ncc_per_channel = cov / (z_std * x_std + eps)  # [B*N, C]
        ncc = ncc_per_channel.mean(dim=1)  # [B*N]

        return ncc

    def get_name(self):
        return self.encoder.name
