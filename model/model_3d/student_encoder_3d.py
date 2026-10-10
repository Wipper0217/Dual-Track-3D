import torch
import torch.nn as nn

from model.model_3d.resnet.basic_block_encoder_3d import Encoder3D, BasicBlockEnc3D


class StudentEncoder3D(nn.Module):
    """
    Student network: 3D ResNet18 encoder
    The architecture is identical to the encoder of ResNetAutoEncoder3D-18
    Input:  [B, 1, 32, 32, 32]
    Output: [B, 256] feature vector
    """
    def __init__(self, in_channels=1):
        super(StudentEncoder3D, self).__init__()
        self.name = "StudentEncoder3D"
        # Reuse the existing Encoder3D; layers=[2,2,2,2] corresponds to ResNet18
        self.encoder = Encoder3D(
            block=BasicBlockEnc3D,
            layers=[2, 2, 2, 2],
            in_channels=in_channels
        )

    def features(self, x):
        """Extract a 256-dim feature vector (calls the mlp=True branch of the encoder)"""
        return self.encoder(x, mlp=True)  # [B, 256]

    def forward(self, x):
        return self.features(x)

    def forward_compile(self, x):
        """torch.compile-compatible forward pass."""
        return self.encoder.forward_compile(x)

    def get_name(self):
        return self.name
