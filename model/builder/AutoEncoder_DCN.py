import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dcn import DCNv1

class AutoEncoder(nn.Module):
    def __init__(self, normalize, in_channels):
        super(AutoEncoder, self).__init__()
        self.normalize = normalize
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = DecoderV2(output_channels=in_channels)
        self.criterion = nn.MSELoss

    def features(self, x, mlp=True):
        return self.encoder(x)

    def get_in_channels(self):
        return self.in_channels

    def forward(self, x):
        print(f"[DEBUG] Forward called! Input shape: {x.shape}, device: {x.device}")
        print("[DEBUG] AutoEncoder input max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        x_ = self.encoder(x)
        x = self.decoder(x_)
        print("[DEBUG] AutoEncoder output max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        return x

class AutoEncoderV1(nn.Module):
    def __init__(self, normalize, in_channels):
        super(AutoEncoderV1, self).__init__()
        self.name = "AutoEncoderV1"
        self.normalize = normalize
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = DecoderV1(normalize=self.normalize, output_channels=in_channels)
        self.criterion = nn.MSELoss

    def features(self, x, mlp=True):
        return self.encoder(x)

    def get_in_channels(self):
        return self.in_channels

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def forward(self, x):
        print(f"[DEBUG] Forward called! Input shape: {x.shape}, device: {x.device}")
        print("[DEBUG] AutoEncoder input max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        x_ = self.encoder(x)
        x = self.decoder(x_)
        print("[DEBUG] AutoEncoder output max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        return x

class AutoEncoderV1Normalized(nn.Module):
    def __init__(self, in_channels):
        super(AutoEncoderV1Normalized, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = DecoderV1Normalized(output_channels=in_channels)
        self.criterion = nn.MSELoss

    def features(self, x):
        return self.encoder(x)

    def get_in_channels(self):
        return self.in_channels

    def forward(self, x):
        print(f"[DEBUG] Forward called! Input shape: {x.shape}, device: {x.device}")
        print("[DEBUG] AutoEncoder input max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        x_ = self.encoder(x)
        x = self.decoder(x_)
        print("[DEBUG] AutoEncoder output max: {:.2f}, min: {:.2f}".format(x.max(), x.min()))
        return x

class AutoEncoderV2(nn.Module):
    def __init__(self, normalize, in_channels):
        super(AutoEncoderV2, self).__init__()
        self.name = "AutoEncoderV2"
        self.normalize = normalize
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = DecoderV2(normalize=normalize, output_channels=in_channels)
        self.criterion = nn.MSELoss

    def features(self, x):
        return self.encoder(x)

    def get_in_channels(self):
        return self.in_channels

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def forward(self, x):
        print(f"[DEBUG] Forward called! Input shape: {x.shape}, device: {x.device}")
        x_ = self.encoder(x)
        x = self.decoder(x_)
        return x


# class Encoder(nn.Module):
#     def __init__(self, in_channels=3):
#         super(Encoder, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),
#         )
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#
#     def forward(self, x):
#         # print("Encoder input: ", x.shape)
#         x = self.block1(x)
#         # print("Encoder block 1 output: ", x.shape)
#         x = self.block2(x)
#         # print("Encoder block 2 output: ", x.shape)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         # print("Encoder fc 1 output: ", x.shape)
#         x = self.fc2(x)
#         # print("Encoder fc 2 output: ", x.shape)
#         x = self.fc3(x)
#         return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            DCNv1(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.block2 = nn.Sequential(
            DCNv1(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        #print("Encoder input: ", x.shape)
        x = self.block1(x)
        #print("Encoder block 1 output: ", x.shape)
        x = self.block2(x)
        #print("Encoder block 2 output: ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #print("Encoder fc 1 output: ", x.shape)
        x = self.fc2(x)
        #print("Encoder fc 2 output: ", x.shape)
        x = self.fc3(x)
        return x

class DecoderV1(nn.Module):
    def __init__(self, normalize, output_channels=3):
        super(DecoderV1, self).__init__()
        self.normalize = normalize
        self.output_channels = output_channels

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128 * 4 * 4)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCNv1(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DCNv1(32, output_channels, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("Decoder input: ", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # print("Decoder fc output: ", x.shape)
        x = x.view(x.size(0), 128, 4, 4)
        x = F.interpolate(x, size=(10, 10), mode='bilinear')
        x = self.block1(x)
        # print("Decoder block 1 output: ", x.shape)
        x = F.interpolate(x, size=(24, 24), mode='bilinear')
        x = self.block2(x)
        if self.normalize:
            x = self.sigmoid(x)
        # print("Decoder block 2 output: ", x.shape)
        return x


class DecoderV1Normalized(nn.Module):
    def __init__(self, output_channels=3):
        self.output_channels = output_channels
        super(DecoderV1Normalized, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128 * 4 * 4)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCNv1(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DCNv1(32, output_channels, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("Decoder input: ", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #print("Decoder fc output: ", x.shape)
        x = x.view(x.size(0), 128, 4, 4)
        x = F.interpolate(x, size=(10, 10), mode='bilinear')
        x = self.block1(x)
        #print("Decoder block 1 output: ", x.shape)
        x = F.interpolate(x, size=(24, 24), mode='bilinear')
        x = self.block2(x)
        x = self.sigmoid(x)
        #print("Decoder block 2 output: ", x.shape)
        return x

class DecoderV2(nn.Module):
    def __init__(self, normalize, output_channels=3):
        super(DecoderV2, self).__init__()
        self.normalize = normalize
        self.output_channels = output_channels
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128 * 4 * 4)

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=0, output_padding=0),
        )

    def forward(self, x):
        print("Decoder input: ", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        print("Decoder fc output: ", x.shape)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.block1(x)
        print("Decoder block 1 output: ", x.shape)
        x = self.block2(x)
        if self.normalize:
            x = self.sigmoid(x)
        return x

class DecoderV3(nn.Module):
    def __init__(self, output_channels=3):
        super(DecoderV3, self).__init__()
        self.output_channels = output_channels
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128 * 4 * 4)

        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=2),
        )

    def forward(self, x):
        print("Decoder input: ", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        print("Decoder fc output: ", x.shape)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.block1(x)
        print("Decoder block 1 output: ", x.shape)
        x = self.block2(x)
        return x
if __name__ == '__main__':
    # encoder = Encoder()
    x = torch.randn(16, 24, 24, 24)
    # xf = encoder(x)
    # print(xf.shape)
    #
    # decoder = Decoder()
    # xx = decoder(xf)
    # print(xx.shape)

    autoencoder = AutoEncoderV1(normalize=True, in_channels=24)
    print(autoencoder(x).shape)

