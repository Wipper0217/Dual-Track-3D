import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

class SDCF(nn.Module):
    def __init__(self, yf, lambda0):
        super(SDCF, self).__init__()
        self.yf = yf.clone()
        self.lambda0 = lambda0

    def forward(self, zf, xf):
        zf_fft = fft.rfft2(zf)
        xf_fft = fft.rfft2(xf)

        # Compute kzzf (autocorrelation); keep the dimensions matched
        kzzf = torch.sum(zf_fft * torch.conj(zf_fft), dim=1, keepdim=True)  # [B, 1, H, W/2+1]

        # Compute kxzf (cross-correlation)
        kxzf = torch.sum(xf_fft * torch.conj(zf_fft), dim=1, keepdim=True)  # [B, 1, H, W/2+1]

        # Compute alpha (filter coefficients)
        alphaf = self.yf / (kzzf + self.lambda0)

        # Obtain the response map (in the time domain)
        response = torch.fft.irfft2(kxzf * alphaf, s=xf.shape[-2:])

        return response



if __name__ == '__main__':

    def create_gaussian_shaped_labels(sz, sigma=0.5):
        x, y = np.meshgrid(np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2),
                           np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2))
        d = x ** 2 + y ** 2
        g = np.exp(-0.5 / (sigma ** 2) * d)
        g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
        g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
        return g.astype(np.float32)

    size = 119
    batch = 3
    xf = torch.randn(batch, 32, size, size)
    zf = torch.randn(batch, 32, size, size)
    y = create_gaussian_shaped_labels([size, size], 0.5)
    y_tensor = torch.from_numpy(y).view(1, 1, size, size)
    yf = fft.rfft2(y_tensor)
    print(yf.shape)
    sdcf = SDCF(yf=yf, lambda0=1e-4)
    print(sdcf(zf, xf).shape)

