import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_3d.resnet.basic_block_encoder_3d import Encoder3D, BasicBlockEnc3D
from model.model_3d.resnet.basic_block_decoder_3d import Decoder3D, BasicBlockDec3D
class ResNetAutoEncoder3D(nn.Module):
    def __init__(self, normalize=True, in_channels=1, num_layers=18):
        super(ResNetAutoEncoder3D, self).__init__()
        self.name = "ResNetAutoEncoder3D"
        self.in_channels = in_channels
        if num_layers == 18:
            self.encoder = Encoder3D(BasicBlockEnc3D, [2, 2, 2, 2], in_channels)
            self.decoder = Decoder3D(BasicBlockDec3D, [2, 2, 2, 2], in_channels)
        elif num_layers == 34:
            self.encoder = Encoder3D(BasicBlockEnc3D,  [3, 4, 6, 3], in_channels)
            self.decoder = Decoder3D(BasicBlockDec3D,  [3, 4, 6, 3], in_channels)
        else:
            raise NotImplementedError("Only resnet_other 18 & 34 autoencoder have been implemented for images size >= 64x64.")
        self.normalize = normalize
        if self.normalize:
            self.sigmoid = nn.Sigmoid()

    def get_name(self):
        return self.name

    def get_in_channels(self):
        return self.in_channels

    def features(self, x, mlp=False):
        return self.encoder(x, mlp)

    def forward(self, x, mlp):
        laten = self.features(x, mlp)
        output = self.decoder(laten)
        if self.normalize:
            output = self.sigmoid(output)
        return output