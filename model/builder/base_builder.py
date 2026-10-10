import torch
import torch.nn as nn


class Base2DAutoEncoder(nn.Module):
    def __init__(self, normalize, in_channels, encoder, decoder):
        super(Base2DAutoEncoder, self).__init__()
        self.normalize = normalize
        self.in_channels = in_channels

        self.encoder = encoder(in_channels)
        self.decoder = decoder(normalize=normalize, output_channels=in_channels)

    def get_name(self):
        raise NotImplementedError

    def feature(self, x):
        return self.encoder(x)

    def get_in_channels(self):
        return self.in_channels

    def forward(self, x):
        feature = self.feature(x)
        return self.decoder(feature)



class Base2DEncoder(nn.Module):
    def __init__(self, in_channels):
        super(Base2DEncoder, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        raise NotImplementedError


class Base2DDecoder(nn.Module):
    def __init__(self, normalize, output_channels):
        super(Base2DDecoder, self).__init__()
        self.normalize = normalize
        self.output_channels = output_channels

    def forward(self, x):
        raise NotImplementedError