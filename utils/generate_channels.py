import numpy as np
from scipy.ndimage import map_coordinates


def generate_channels(image, center, channels=3, channel_24_direction=None):
    center = [int(i) for i in center]
    if channels == 3:
        return generate_channels_3(image, center)
    elif channels == 9:
        return generate_channels_9(image, center)
    elif channels == 24:
        if channel_24_direction is None:
            channel_24_direction = 'z'
        if channel_24_direction == 'x':
            return generate_channels_24_x(image, center)
        elif channel_24_direction == 'y':
            return generate_channels_24_y(image, center)
        else:
            return generate_channels_24_z(image, center)


def generate_channels_3(image, center):
    """
    :param image: (z, y, x)
    :return: 3 channels
    """
    cx, cy, cz = center
    patch_axial = image[cz, (cy-12):(cy+12), (cx-12):(cx+12)] # x-y
    patch_coronal = image[(cz-12):(cz+12), cy, (cx-12):(cx+12)] # z-x
    patch_sagittal = image[(cz-12):(cz+12), (cy-12):(cy+12), cx]  # z-y
    crop = np.stack([patch_axial, patch_coronal, patch_sagittal], axis=0)
    return crop

def generate_channels_9(image, center, output_size=24):
    D, H, W = image.shape
    out_h, out_w = output_size, output_size

    # --- 1. Determine the center point ---
    cx, cy, cz = center
    center = np.array([cz, cy, cx])
    # --- 2. Generate relative grid offsets (Relative Grid Offsets) ---
    # What is generated here is the offset relative to (0,0), e.g. [-12, -11, ..., 0, ..., 11]
    # grid_y corresponds to the slice height direction (v vector)
    # grid_x corresponds to the slice width direction (u vector)
    grid_y, grid_x = np.meshgrid(
        np.arange(out_h) - (out_h / 2 - 0.5),
        np.arange(out_w) - (out_w / 2 - 0.5),
        indexing='ij'
    )

    slices = []

    # Auxiliary constant: sin(45) = cos(45) = 1/sqrt(2)
    s = np.sqrt(2) / 2

    # --- 3. Define the direction vectors of the 9 planes (Configuration) ---
    # Format: dict(u=[z, y, x], v=[z, y, x])
    # u: direction to the right of the slice image
    # v: direction downward of the slice image

    configs = []

    # --- Group 1: 3 orthogonal planes ---
    # We also use the vector-based approach to support interpolation for arbitrary floating-point center
    configs.append(dict(name="Axial(XY)", u=[0, 0, 1], v=[0, 1, 0]))  # Z unchanged
    configs.append(dict(name="Coronal(XZ)", u=[0, 0, 1], v=[1, 0, 0]))  # Y unchanged
    configs.append(dict(name="Sagittal(YZ)", u=[0, 1, 0], v=[1, 0, 0]))  # X unchanged

    # --- Group 2: 2 rotations around the Z axis (Vertical Diagonals) ---
    # Keep the Z axis (v=[1,0,0]) unchanged, rotate u in the XY plane
    configs.append(dict(name="Z-Rot-Pos", u=[0, s, s], v=[1, 0, 0]))
    configs.append(dict(name="Z-Rot-Neg", u=[0, -s, s], v=[1, 0, 0]))

    # --- Group 3: 2 rotations around the Y axis (Front-Back Diagonals) ---
    # Keep the Y axis (u=[0,1,0]) unchanged, rotate v in the XZ plane
    configs.append(dict(name="Y-Rot-Pos", u=[0, 1, 0], v=[s, 0, s]))
    configs.append(dict(name="Y-Rot-Neg", u=[0, 1, 0], v=[-s, 0, s]))

    # --- Group 4: 2 rotations around the X axis (Side-Side Diagonals) ---
    # Keep the X axis (u=[0,0,1]) unchanged, rotate v in the YZ plane
    configs.append(dict(name="X-Rot-Pos", u=[0, 0, 1], v=[s, s, 0]))
    configs.append(dict(name="X-Rot-Neg", u=[0, 0, 1], v=[-s, s, 0]))

    # --- 4. Unified sampling ---
    for cfg in configs:
        u_vec = np.array(cfg['u'])
        v_vec = np.array(cfg['v'])

        # Compute the 3D absolute coordinates
        # Formula: P_absolute = Center + (offset_y * v) + (offset_x * u)

        # Component computation (broadcast mechanism)
        p_z = center[0] + grid_y * v_vec[0] + grid_x * u_vec[0]
        p_y = center[1] + grid_y * v_vec[1] + grid_x * u_vec[1]
        p_x = center[2] + grid_y * v_vec[2] + grid_x * u_vec[2]

        coords = np.stack([p_z, p_y, p_x])

        # Interpolated sampling
        # mode='constant', cval=0.0 handles the out-of-bounds case
        patch = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
        slices.append(patch)

    return np.stack(slices, axis=0)


def crop_3d(image, center, img_size=24):
    center = [int(i) for i in center]
    r = img_size // 2
    cx, cy, cz = center
    crop = image[cz - r:cz + r, cy - r:cy + r, cx - r:cx + r]
    return crop[np.newaxis, ...]

def generate_channels_24(image, center, img_size=24):
    """

    :param image: (z, y, x)
    :param center: (cx, cy, cz)
    :return:
    """
    center = [int(i) for i in center]
    r = img_size // 2
    cx, cy, cz = center
    crop = image[cz-r:cz+r, cy-r:cy+r, cx-r:cx+r]
    return crop

def generate_channels_24_x(image, center, img_size=24):
    crop_channels_24 = generate_channels_24(image, center, img_size)
    crop_channels_24_x = crop_channels_24.transpose(2, 0, 1).astype(np.float32) # (x, z, y)
    return crop_channels_24_x

def generate_channels_24_y(image, center, img_size=24):
    crop_channels_24 = generate_channels_24(image, center, img_size)
    crop_channels_24_y = crop_channels_24.transpose(1, 0, 2).astype(np.float32) # (y, z, x)
    return crop_channels_24_y

def generate_channels_24_z(image, center, img_size=24):
    crop_channels_24 = generate_channels_24(image, center, img_size)
    crop_channels_24_z = crop_channels_24.astype(np.float32) #  (z, y, x)
    return crop_channels_24_z


def get_generate_channels_function(method, in_channels, channels_24_direction=None):
    if method == '2d':
        if in_channels == 3:
            return generate_channels_3
        elif in_channels == 9:
            return generate_channels_9
        else:
            if channels_24_direction is None:
                channels_24_direction = 'z'

            if channels_24_direction == 'x':
                return generate_channels_24_x
            elif channels_24_direction == 'y':
                return generate_channels_24_y
            else:
                return generate_channels_24_z
    else:
        return crop_3d

if __name__ == '__main__':
    # Create simple 24x24x24 data
    vol = np.random.rand(200, 200, 200)

    # Case A: no center specified (use the default center)
    out1 = generate_channels_9(vol, center=[100, 80, 111], output_size=24)
    print(f"Default Center Shape: {out1.shape}")

    # Case B: center with a specified offset (e.g. focus on some lesion point [5, 12, 12])
    # Even if the center is at the edge, the code automatically pads with 0 (padding)
    out2 = generate_channels_9(vol, center=[5, 12, 12])
    print(f"Custom Center Shape:  {out2.shape}")