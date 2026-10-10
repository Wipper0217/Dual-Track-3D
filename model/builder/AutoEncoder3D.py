import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder3D(nn.Module):
    def __init__(self, normalize):
        super(AutoEncoder3D, self).__init__()
        self.name = "ConvAutoEncoder3D"
        self.encoder = Encoder3D()
        self.decoder = Decoder3D(normalize)

    def get_name(self):
        return self.name

    def get_in_channels(self):
        return 24

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        x_ = self.encoder(x)
        x = self.decoder(x_)
        return x


class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
        )

        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        #self.fc4 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, normalize):
        super(Decoder3D, self).__init__()

        self.normalize = normalize

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 128 * 4 * 4 * 4)

        self.block1 = nn.Sequential(
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 3, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(x.size(0), 128, 4, 4, 4)
        x = F.interpolate(x, size=(10, 10, 10), mode='trilinear', align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, size=(24, 24, 24), mode='trilinear', align_corners=False)
        x = self.block2(x)
        if self.normalize:
            x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    model = AutoEncoder3D(True)
    x = torch.randn(1, 1, 24, 24, 24)
    y = model(x)
    print(y.shape)