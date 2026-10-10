import numpy as np
from contrastive_learning.dataset.crop_utils import random_crop_numpy
class Augmentation:
    def __init__(self, output_size, radius, scale_factor=0.0, context_amount=0.5):
        self.output_size = output_size
        self.radius = radius
        self.scale_factor = scale_factor
        self.context_amount = context_amount
        self.rot_k = [1, 2, 3, 4]
        self.rot_axes = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def _rotate(self, img):
        k = np.random.choice(self.rot_k)
        axes = self.rot_axes[np.random.choice(len(self.rot_axes))]
        return np.rot90(img, k=k, axes=axes)

    def _crop(self, img, center, size):
        crop = random_crop_numpy(img, center, size, self.output_size, self.radius, self.scale_factor, self.context_amount)
        return self._rotate(crop)
    def __call__(self, img, center, size):
        return self._crop(img, center, size)

class MoCoAugment(object):
    def __init__(self, output_size, radius, scale_factor=0.0, context_amount=0.5):
        self.output_size = output_size
        self.radius = radius
        self.scale_factor = scale_factor
        self.context_amount = context_amount
        self.rot_k = [1, 2, 3, 4]
        self.rot_axes = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

if __name__ == '__main__':
    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    file_path = r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D\EMC-01\Data\001.mha"
    img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    plt.imshow(img[52, :, :], cmap='gray')
    plt.title("original")
    plt.show()
    print(img.shape)

    flipped_z = np.flip(img, axis=0)
    plt.imshow(flipped_z[52, :, :], cmap='gray')
    plt.title("flip z")
    plt.show()
    flipped_y = np.flip(img, axis=1)
    plt.imshow(flipped_y[52, :, :], cmap='gray')
    plt.title("flip y")
    plt.show()
    flipped_x = np.flip(img, axis=2)
    plt.imshow(flipped_x[52, :, :], cmap='gray')
    plt.title("flip x")
    plt.show()
    print(flipped_z.shape)
    print(flipped_y.shape)
    print(flipped_x.shape)

    rot_90 = np.rot90(img, k=1, axes=(1, 2))
    plt.imshow(rot_90[52, :, :], cmap='gray')
    plt.title("rotate 90")
    plt.show()
    print(rot_90.shape)