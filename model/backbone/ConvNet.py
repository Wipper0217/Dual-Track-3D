import torch
from torch import nn


class DCFNetFeature(nn.Module):

    def __init__(self):
        super(DCFNetFeature, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        # self.conv2 = nn.Conv2d(32, 32, 3)
        # self.conv3 = nn.Conv2d(32, 32, 3)
        # self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
        )

    def forward(self, x):
        x = self.feature(x)
        return x


if __name__ == '__main__':

    x = torch.rand(1, 3, 125, 125)
    net = DCFNetFeature()
    f = net(x)
    print(f.shape)
