import numpy as np
from matplotlib import pyplot as plt


def create_gaussian_label(size, sigma, normalize=False):
    h, w = size
    shape = (h, w)

    # If no center point is provided, default to the center of the image
    center = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)

    # Create a coordinate grid centered on the center
    y, x = np.ogrid[:shape[0], :shape[1]]
    y = y - center[0]  # row index (y coordinate)
    x = x - center[1]  # column index (x coordinate)

    # Compute the Gaussian function
    map = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    # map[map < np.finfo(map.dtype).eps * map.max()] = 0

    # Normalization (optional)
    if normalize:
        sumh = map.sum()
        if sumh != 0:
            map /= sumh
    # map = map.reshape((1, 1, h, w))
    # map = np.tile(map, (b, c, 1, 1))
    # map = torch.from_numpy(map).to(self.device).float()
    return map

if __name__ == '__main__':
    size = (119, 119)
    sigma = 0.5
    label1 = create_gaussian_label(size, sigma)
    label2 = create_gaussian_label(size, sigma*2)

    plt.imshow(label1)
    plt.colorbar()
    plt.title("label 1")
    plt.show()

    plt.imshow(label2)
    plt.colorbar()
    plt.title("label 2")
    plt.show()