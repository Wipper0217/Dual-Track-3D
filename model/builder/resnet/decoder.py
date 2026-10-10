import math

import torch
from torch import nn

from model.builder.resnet.block import BasicBlockDec


class Decoder(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        super(Decoder, self).__init__()
        self.inplanes = 512 * block.expansion

        self.layer1 = self._make_layer(block, 512, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)


        self.layer3 = self._make_layer(block, 128, layers[2],
                                       stride=1, dilation=2)  # 15x15, 7x7
        self.feature_size = (256 + 128) * block.expansion
        self.layer4 = self._make_layer(block, 64, layers[3],
                                       stride=1, dilation=4)  # 7x7, 3x3
        self.feature_size = 512 * block.expansion

        self.conv1 = nn.Conv2d(64, in_channels, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,  # kernel=3
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)

        x = self.conv1(p4)

        return x

if __name__ == '__main__':
    model = Decoder(BasicBlockDec, [2, 2, 2, 2])
    x = torch.randn(3, 512, 2, 2)
    y = model(x)
    print(y.shape)