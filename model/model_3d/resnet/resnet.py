import torch
from torch import nn
from torch import Tensor

class ConvBlock(nn.Module):
    """Convolution block for downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvTransposeBlock(nn.Module):
    """Convolution transpose block for upsampling."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ResNetBlock(nn.Module):
    """3D ResNet block which preserves the number of input channels."""
    def __init__(self, n_channels, kernel_size, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(n_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(n_channels, n_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(n_channels)
        )

    def forward(self, x):
        out = self.conv(x) + x
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNetBlock_v2(nn.Module):
    """3D ResNet_v2 block which preserves the number of input channels."""
    def __init__(self, n_channels, kernel_size, stride=1, padding=1):
        super(ResNetBlock_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size, stride, padding, bias=False),

            nn.BatchNorm3d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size, stride, padding, bias=False)
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out

class Encoder(nn.Module):
    def __init__(self, layers, patch_size, latent_dim, n_channels, hidden_dim, res_block=ResNetBlock):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = ConvBlock(n_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.res_block1 = self._make_layer(res_block, 64, layers[0])
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.res_block2 = self._make_layer(res_block, 128, layers[1])
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.res_block3 = self._make_layer(res_block, 256, layers[2])
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.res_block4 = self._make_layer(res_block, 512, layers[3])
        self.fc1 = nn.Linear(512 * int(patch_size/16) * int(patch_size/16) * int(patch_size/16), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def _make_layer(self, block, n_channels, n_layers):
        layers = []
        layers.append(
            block(n_channels, kernel_size=3)
        )
        for _ in range(1, n_layers):
            layers.append(
                block(n_channels, kernel_size=3)
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.conv2(x)
        x = self.res_block2(x)
        x = self.conv3(x)
        x = self.res_block3(x)
        x = self.conv4(x)
        x = self.res_block4(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        mu = self.fc2(x)
        log_sigma = self.fc3(x)
        N = torch.normal(size=(1, self.latent_dim), mean=0.0, std=1.0)
        z = mu + torch.exp(0.5 * log_sigma) * N.type_as(mu)

        return z, mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, layers, patch_size, latent_dim, n_channels, hidden_dim, res_block=ResNetBlock):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 512 * int(patch_size/16) * int(patch_size/16) * int(patch_size/16))
        self.relu = nn.ReLU(inplace=True)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, int(patch_size/16), int(patch_size/16), int(patch_size/16)))
        self.conv1 = ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=4, stride=2)
        self.res_block1 = self._make_layer(res_block, 256, layers[0])
        self.conv2 = ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2)
        self.res_block2 = self._make_layer(res_block, 128, layers[1])
        self.conv3 = ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2)
        self.res_block3 = self._make_layer(res_block, 64, layers[2])
        self.conv4 = ConvTransposeBlock(in_channels=64, out_channels=n_channels, kernel_size=4, stride=2)
        self.res_block4 = self._make_layer(res_block, n_channels, layers[3])
        self.conv5 = nn.ConvTranspose3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, n_channels, n_layers):
        layers = []
        layers.append(
            block(n_channels, kernel_size=3)
        )
        for _ in range(1, n_layers):
            layers.append(
                block(n_channels, kernel_size=3)
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.unflatten(x)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.conv2(x)
        x = self.res_block2(x)
        x = self.conv3(x)
        x = self.res_block3(x)
        x = self.conv4(x)
        x = self.res_block4(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x


class ResNet12_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet12_3D, self).__init__()
        self.encoder = Encoder([1, 1, 1, 1], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)
        self.decoder = Decoder([1, 1, 1, 1], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z


class ResNet12v2_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet12v2_3D, self).__init__()
        self.encoder = Encoder([1, 1, 1, 1], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)
        self.decoder = Decoder([1, 1, 1, 1], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z


class ResNet18_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet18_3D, self).__init__()
        self.encoder = Encoder([2, 2, 2, 2], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)
        self.decoder = Decoder([2, 2, 2, 2], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z


class ResNet18v2_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet18v2_3D, self).__init__()
        self.encoder = Encoder([2, 2, 2, 2], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)
        self.decoder = Decoder([2, 2, 2, 2], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z


class ResNet34_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet34_3D, self).__init__()
        self.encoder = Encoder([3, 4, 6, 3], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)
        self.decoder = Decoder([3, 4, 6, 3], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z


class ResNet34v2_3D(nn.Module):
    def __init__(self, patch_size, latent_dim=128, n_channels=3, hidden_dim=8192):
        super(ResNet34v2_3D, self).__init__()
        self.encoder = Encoder([3, 4, 6, 3], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)
        self.decoder = Decoder([3, 4, 6, 3], patch_size, latent_dim, n_channels, hidden_dim, ResNetBlock_v2)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z

if __name__ == '__main__':
    model = ResNet18_3D(patch_size=32, latent_dim=128, n_channels=1, hidden_dim=8192)
    x = torch.randn(3, 1, 32, 32, 32)
    x_hat, mu, log_sigma, z = model(x)
    print(x_hat.shape)
    print(mu.shape)
    print(log_sigma.shape)
    print(z.shape)