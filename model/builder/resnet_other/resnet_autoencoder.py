import torch
import torch.nn as nn

from model.builder.resnet_other.basic_block_decoder import Decoder, BasicBlockDec
from model.builder.resnet_other.basic_block_encoder import Encoder, BasicBlockEnc
from model.builder.resnet_other.light_basic_block_decoder import LightDecoder, LightBasicBlockDec
from model.builder.resnet_other.light_basic_block_encoder import LightEncoder, LightBasicBlockEnc


class ResNetAutoEncoder(nn.Module):
    """Construction of resnet_other autoencoder.

        Attributes:
            network (str): the architectural type of the network. There are 2 choices:
                - 'default' (default), related with the original resnet_other-18 architecture
                - 'light', a smaller network implementation of resnet_other-18 for smaller input images.
            num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types
                of network, 34 layers for default only network and 20 layers for light network.
        """

    def __init__(self, in_channels=3, network='default', num_layers=18):
        """Initialize the autoencoder.

               Args:
                   network (str): a flag to define the network version. Choices ['default' (default), 'light'].
                    num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for
                       'default' network), 20 (only for 'light' network).
               """
        super().__init__()
        self.network = network
        self.in_channels = in_channels
        if self.network == 'default':
            if num_layers == 18:
                # resnet_other 18 encoder
                self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2], in_channels)
                # resnet_other 18 decoder
                self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2], in_channels)
            elif num_layers == 34:
                # resnet_other 34 encoder
                self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3], in_channels)
                # resnet_other 34 decoder
                self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3], in_channels)
            else:
                raise NotImplementedError(
                    "Only resnet_other 18 & 34 autoencoder have been implemented for images size >= 64x64.")
        elif self.network == 'light':
            if num_layers == 18:
                # resnet_other 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [2, 2, 2])
                # resnet_other 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [2, 2, 2])
            elif num_layers == 20:
                # resnet_other 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [3, 3, 3])
                # resnet_other 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [3, 3, 3])
            else:
                raise NotImplementedError(
                    "Only resnet_other 18 & 20 autoencoder have been implemented for images size < 64x64.")
        else:
            raise NotImplementedError(
                "Only default and light resnet_other have been implemented. Th light version corresponds to input datasets with size less than 64x64.")

    def get_name(self):
        return "ResNetAutoEncoder"

    def get_in_channels(self):
        """Get the number of input channels of the model.

        Returns:
            int: number of input channels
        """
        return self.in_channels

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): the batched input data

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        z = self.encoder(x)

        x = self.decoder(z)
        return x


if __name__ == '__main__':
    model = ResNetAutoEncoder(in_channels=24)
    x = torch.randn(16, 24, 24, 24)
    y = model(x)
    print(y.shape)