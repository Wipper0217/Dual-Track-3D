import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class BasicBlockEnc(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlockEnc, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
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


class BasicBlockDec(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 upsample=None, dilation=1):
        """
        The corresponding Decoder module.
        - inplanes: number of input channels (e.g. 512)
        - planes: number of output channels (e.g. 256)
        - stride: upsampling stride, usually 2
        - upsample: the _upsample layer used for the residual connection (corresponds to the Encoder's downsample)
        """
        super(BasicBlockDec, self).__init__()

        # Compute output_padding for ConvTranspose2d
        # When stride=2, output_padding=1 ensures H_out = 2 * H_in
        # When stride=1, output_padding=0 ensures H_out = H_in
        output_padding = 1 if stride > 1 else 0

        # First convolution layer of the main path, used for upsampling and changing the number of channels
        # (inplanes -> planes, H -> H*stride)
        self.conv1 = nn.ConvTranspose2d(inplanes, planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,  # padding=1, kernel=3, stride=2, out_padding=1
                                        output_padding=output_padding,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer of the main path, used for feature refinement (does not change the size)
        # (planes -> planes, H*stride -> H*stride)
        # We assume conv3x3 in the Decoder uses the standard dilation=1
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        # Upsampling module for the residual path
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        # Keep the residual
        residual = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual path
        # If an upsample module is defined (i.e. the number of channels or the size of input and output do not match),
        # upsample the original residual x
        if self.upsample is not None:
            residual = self.upsample(x)

        # Add the residual
        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    x = torch.randn(3, 64, 5, 5)
    # model = BasicBlockEnc(64, 64, stride=2)
    # y = model(x)
    # print(y.shape)
    # decoder = BasicBlockDec(64, 64, stride=2)
    # z = decoder(y)
    # print(z.shape)