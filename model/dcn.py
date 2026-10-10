from torch import nn
from torchvision.ops import DeformConv2d


class DCNv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(DCNv1, self).__init__()
        # The number of offset output channels is 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True # offset prediction usually requires a bias
        )

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Initialize the weight and bias of offset_conv to 0 so that the initial behavior is close to standard convolution
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.dcn(x, offset)
        return out