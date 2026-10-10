import os

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
def crop_and_resize_torch(img, center, size, output_size, scale_factor=0.0, context_amount=0.5):
    D, H, W = img.shape
    cx, cy, cz = center
    # add random scale variation
    if scale_factor > 0:
        scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
        size = [int(s * scale * (1 + context_amount)) for s in size]
    else:
        size = [int(size[0] * (1 + context_amount)),
                int(size[1] * (1 + context_amount)),
                int(size[2] * (1 + context_amount))]
    d, h, w = size

    z0 = int(round(cz - d / 2))
    y0 = int(round(cy - h / 2))
    x0 = int(round(cx - w / 2))

    z1 = z0 + d
    y1 = y0 + h
    x1 = x0 + w

    # pad_z0 = max(0, -z0)
    # pad_y0 = max(0, -y0)
    # pad_x0 = max(0, -x0)
    # pad_z1 = max(0, z1 - D)
    # pad_y1 = max(0, y1 - H)
    # pad_x1 = max(0, x1 - W)

    z0_clip = max(0, z0)
    y0_clip = max(0, y0)
    x0_clip = max(0, x0)
    z1_clip = min(D, z1)
    y1_clip = min(H, y1)
    x1_clip = min(W, x1)

    cropped = img[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
    cropped_tensor = torch.from_numpy(cropped).float()
    # add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    cropped_tensor = cropped_tensor.unsqueeze(0).unsqueeze(0)
    # use trilinear interpolation to resize the image to (output_size, output_size, output_size)
    resized = F.interpolate(cropped_tensor, size=(output_size, output_size, output_size), mode='trilinear',
                            align_corners=False)

    return resized

def crop_and_resize_numpy(img, center, size, output_size, scale_factor=0.0, context_amount=0.5):
    D, H, W = img.shape
    cx, cy, cz = center

    # add random scale variation
    if scale_factor > 0:
        scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
        size = [int(s * scale * (1 + context_amount)) for s in size]
    else:
        size = [int(size[0] * (1 + context_amount)),
                int(size[1] * (1 + context_amount)),
                int(size[2] * (1 + context_amount))]
    d, h, w = size

    z0 = int(round(cz - d / 2))
    y0 = int(round(cy - h / 2))
    x0 = int(round(cx - w / 2))

    z1 = z0 + d
    y1 = y0 + h
    x1 = x0 + w

    # pad_z0 = max(0, -z0)
    # pad_y0 = max(0, -y0)
    # pad_x0 = max(0, -x0)
    # pad_z1 = max(0, z1 - D)
    # pad_y1 = max(0, y1 - H)
    # pad_x1 = max(0, x1 - W)

    z0_clip = max(0, z0)
    y0_clip = max(0, y0)
    x0_clip = max(0, x0)
    z1_clip = min(D, z1)
    y1_clip = min(H, y1)
    x1_clip = min(W, x1)

    cropped = img[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
    zoom_factors = (output_size / d, output_size / h, output_size / w)
    resized = zoom(cropped, zoom_factors, order=1)
    return resized

def crop_numpy(img, center, output_size):
    D, H, W = img.shape
    cx, cy, cz = center
    half_size = output_size // 2

    z0 = int(round(cz - half_size))
    y0 = int(round(cy - half_size))
    x0 = int(round(cx - half_size))

    z1 = int(cz + half_size)
    y1 = int(cy + half_size)
    x1 = int(cx + half_size)


    crop_img = img[z0:z1, y0:y1, x0:x1]
    return crop_img


def crop_numpy_batch(img, centers, output_size, pad_value=0):
    """Batch-crop multiple candidate points, returning a numpy array of shape (N, D, H, W)

    :param img: 3D numpy array (D, H, W)
    :param centers: list of candidate points, each element is [cx, cy, cz]
    :param output_size: crop output size
    :param pad_value: boundary padding value
    :return: numpy array of shape (N, output_size, output_size, output_size)
    """
    D, H, W = img.shape
    half = output_size // 2
    patches = []

    for center in centers:
        cx, cy, cz = center
        z0 = int(round(cz - half))
        y0 = int(round(cy - half))
        x0 = int(round(cx - half))
        z1 = z0 + output_size
        y1 = y0 + output_size
        x1 = x0 + output_size

        pad_z0 = max(0, -z0)
        pad_y0 = max(0, -y0)
        pad_x0 = max(0, -x0)
        pad_z1 = max(0, z1 - D)
        pad_y1 = max(0, y1 - H)
        pad_x1 = max(0, x1 - W)

        if any([pad_z0, pad_y0, pad_x0, pad_z1, pad_y1, pad_x1]):
            img_padded = np.pad(
                img,
                ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                constant_values=pad_value
            )
            z0 += pad_z0
            y0 += pad_y0
            x0 += pad_x0
            patch = img_padded[z0:z0 + output_size, y0:y0 + output_size, x0:x0 + output_size]
        else:
            patch = img[z0:z1, y0:y1, x0:x1]
        patches.append(patch)

    return np.stack(patches, axis=0)


def crop_numpy_safe(img, center, output_size, pad_value=0):
    """Crop with edge padding, guaranteeing the output is strictly (output_size, output_size, output_size)"""
    D, H, W = img.shape
    cx, cy, cz = center
    half = output_size // 2

    z0 = int(round(cz - half))
    y0 = int(round(cy - half))
    x0 = int(round(cx - half))

    z1 = z0 + output_size
    y1 = y0 + output_size
    x1 = x0 + output_size

    pad_z0 = max(0, -z0)
    pad_y0 = max(0, -y0)
    pad_x0 = max(0, -x0)
    pad_z1 = max(0, z1 - D)
    pad_y1 = max(0, y1 - H)
    pad_x1 = max(0, x1 - W)

    if any([pad_z0, pad_y0, pad_x0, pad_z1, pad_y1, pad_x1]):
        img = np.pad(
            img,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            constant_values=pad_value
        )
        z0 += pad_z0
        y0 += pad_y0
        x0 += pad_x0

    return img[z0:z0 + output_size, y0:y0 + output_size, x0:x0 + output_size]


def random_crop_numpy(img, center, size, output_size, radius, scale_factor=0.0, context_amount=0.5):
    cx, cy, cz = center
    d, h, w = size
    half = output_size // 2
    offset_z = np.random.randint(-radius, radius + 1)
    offset_y = np.random.randint(-radius, radius + 1)
    offset_x = np.random.randint(-radius, radius + 1)

    new_cz = cz + offset_z
    new_cy = cy + offset_y
    new_cx = cx + offset_x

    new_cz = int(new_cz)
    new_cy = int(new_cy)
    new_cx = int(new_cx)
    #crop_img = crop_and_resize_numpy(img, center_new, size, output_size, scale_factor, context_amount)
    crop_img = img[new_cz - half:new_cz + half, new_cy - half:new_cy + half, new_cx - half:new_cx + half]
    return crop_img[np.newaxis, ...].astype(np.float32)



if __name__ == '__main__':
    import SimpleITK as sitk
    file_path = r"D:\DataSet\MICCAI_CLUST_2015_TrainingSet_3D\MICCAI_CLUST_2015_TrainingSet_3D\EMC-01\Data\001.mha"
    img_name = os.path.basename(file_path)
    img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    output_size = 24
    half_size = output_size // 2
    center = [126, 114, 52]
    size = [18.0, 18.0, 18.0]
    resize = crop_and_resize_torch(img, center, size, output_size, context_amount=1.0)
    resize_numpy = resize.squeeze(0).squeeze(0).numpy()
    plt.imshow(resize_numpy[half_size, :, :], cmap='gray')
    plt.title("image {}---crop and resize".format(img_name))
    plt.show()

    resize_numpy = crop_and_resize_numpy(img, center, size, output_size, context_amount=1.0)
    plt.imshow(resize_numpy[half_size, :, :], cmap='gray')
    plt.title("crop and resize")
    plt.show()
