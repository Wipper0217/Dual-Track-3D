import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, List, Optional, Tuple


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
              dilation: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    # [3D] Changed to Conv3d
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    # [3D] Changed to Conv3d
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockEnc3D(nn.Module):
    """
    [3D] 3D version of the ResNet BasicBlock
    """
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            # [3D] Changed to BatchNorm3d
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # [3D] Use 3D convolution
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # [3D] Use 3D convolution
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder3D(nn.Module):
    """
    [3D] 3D encoder model
    """

    def __init__(
            self,
            block: Type[BasicBlockEnc3D],  # [3D] Ensure a 3D Block is used
            layers: List[int],
            in_channels=3,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None

    ) -> None:
        super().__init__()
        if norm_layer is None:
            # [3D] Changed to BatchNorm3d
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # [3D] Change Conv2d to Conv3d, and adjust kernel, stride, padding
        # (3, 7, 7) is a common initial kernel size when processing 3D data
        # A stride of (1, 2, 2) downsamples first in the spatial (H, W) dimensions rather than in the depth (D) dimension
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # [3D] Changed to MaxPool3d.
        # k=3, s=2, p=1 will downsample by a factor of 2 along all three dimensions (D, H, W)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        # [3D] stride=2 will be converted to (1, 2, 2) in _make_layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])



        self.fc1 = nn.Linear(256 * block.expansion * 2 * 2 * 2, 1024)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 256)

        for m in self.modules():
            # [3D] Changed to Conv3d
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # [3D] Changed to BatchNorm3d
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                # [3D] Ensure the 3D Block is checked
                if isinstance(m, BasicBlockEnc3D) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[BasicBlockEnc3D],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # [3D] Convert stride (int) to a 3D tuple
        # We assume downsampling only in the H and W dimensions, keeping the D dimension unchanged
        layer_stride = stride
        downsample_stride = stride
        if stride != 1:
            layer_stride = (stride, stride, stride)
            downsample_stride = (stride, stride, stride)

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # [3D] Use 1x1x1 convolution and 3D stride
                conv1x1x1(self.inplanes, planes * block.expansion, downsample_stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                # [3D] Pass in the 3D stride
                self.inplanes, planes, layer_stride, downsample, self.groups, self.base_width, previous_dilation,
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.layer4(x)

        # [3D] Here x.view(x.size(0), -1) automatically handles (B, C, D, H, W) -> (B, C*D*H*W)
        # It correctly flattens (B, 256, 8, 2, 2) into (B, 8192)
        #print(x.shape)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc1_relu(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc3(x)

        return x

    def _inference(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x: Tensor, mlp) -> Tensor:
        if mlp:
            return self._forward_impl(x)
        else:
            return self._inference(x)

if __name__ == '__main__':
    encoder = Encoder3D(block=BasicBlockEnc3D, layers=[2, 2, 2, 2], in_channels=1)
    x = torch.randn(3, 1, 32, 32, 32)
    print(encoder(x, False).shape)