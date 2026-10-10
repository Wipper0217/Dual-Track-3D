import os
import SimpleITK as sitk

import numpy as np
from scipy.ndimage import map_coordinates


def crop_3channels(img, img_size=24):
    d, h, w = img.shape

    p_floor = img_size // 2
    p_ceil = img_size - p_floor

    z_min, z_max = p_floor, d - p_ceil
    y_min, y_max = p_floor, h - p_ceil
    x_min, x_max = p_floor, w - p_ceil



    is_valid = False
    crop_volume = None
    while not is_valid:
        cz = np.random.randint(z_min, z_max)
        cy = np.random.randint(y_min, y_max)
        cx = np.random.randint(x_min, x_max)
        patch_axial = img[cz, (cy - p_floor):(cy + p_ceil), (cx - p_floor):(cx + p_ceil)] # x-y
        if not is_valid_data(patch_axial):
            #print("patch_axial is not valid")
            continue
        patch_coronal = img[(cz - p_floor):(cz + p_ceil), cy, (cx - p_floor):(cx + p_ceil)] # z-x
        if not is_valid_data(patch_coronal):
            #print("patch_coronal is not valid")
            continue
        patch_sagittal = img[(cz - p_floor):(cz + p_ceil), (cy - p_floor):(cy + p_ceil), cx]  # z-y

        if is_valid_data(patch_sagittal):
            is_valid = True
            crop_volume = np.stack([patch_axial, patch_coronal, patch_sagittal], axis=0)
    return crop_volume.astype(np.float32)


def crop_9channels(image, output_size=24, max_retries=100):
    D, H, W = image.shape
    nonzero_mask = image > 0
    nonzero_coords = np.argwhere(nonzero_mask)
    # Determine the boundary of the valid (nonzero) region

    z_min, y_min, x_min = nonzero_coords.min(axis=0)
    z_max, y_max, x_max = nonzero_coords.max(axis=0)

    # --- 2. Prepare the sampling grid ---
    # Generate relative offsets w.r.t. the center point [-11.5, -10.5, ..., 11.5]
    grid_y, grid_x = np.meshgrid(
        np.arange(output_size) - (output_size / 2 - 0.5),
        np.arange(output_size) - (output_size / 2 - 0.5),
        indexing='ij'
    )

    configs = []

    s = np.sqrt(2) / 2
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

    # --- 3. Loop sampling until min > 0 is satisfied ---
    for _ in range(max_retries):
        # Randomly pick a center [cz, cy, cx] within the valid region
        # Suggest shrinking inward by output_size//2 from the boundary to reduce out-of-bounds probability
        margin = output_size // 2
        cz = np.random.uniform(z_min + margin, z_max - margin) if (z_max - z_min) > output_size else (z_min + z_max) / 2
        cy = np.random.uniform(y_min + margin, y_max - margin) if (y_max - y_min) > output_size else (y_min + y_max) / 2
        cx = np.random.uniform(x_min + margin, x_max - margin) if (x_max - x_min) > output_size else (x_min + x_max) / 2

        # Add random jitter
        final_cz, final_cy, final_cx = cz, cy, cx

        temp_slices = []
        is_valid = True

        # Iterate over the 9 plane configurations (configs should be defined outside the function or here)
        for cfg in configs:
            u_vec, v_vec = np.array(cfg['u']), np.array(cfg['v'])

            # Core formula: cz corresponds to D, cy corresponds to H, cx corresponds to W
            p_z = final_cz + grid_y * v_vec[0] + grid_x * u_vec[0]
            p_y = final_cy + grid_y * v_vec[1] + grid_x * u_vec[1]
            p_x = final_cx + grid_y * v_vec[2] + grid_x * u_vec[2]

            coords = np.stack([p_z, p_y, p_x])

            # Use mode='constant', cval=0 to detect edges
            patch = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)

            # Check whether the minimum value is greater than 0
            if np.any(patch <= 0):
                is_valid = False
                break

            temp_slices.append(patch)

        if is_valid and len(temp_slices) == 9:
            # Successfully found a 9-channel image satisfying the requirement
            return np.stack(temp_slices, axis=0).astype(np.float32)

    print(f"Warning: Failed to find valid crop after {max_retries} retries.")
    return None

def crop_9channels_new(image, center=None, radius=12, output_size=24):
    print("image shape={}, radius={}. max={}".format(image.shape, radius, image.max()))
    D, H, W = image.shape
    out_h, out_w = output_size, output_size

    # --- 1. Determine the center point ---
    if center is None:
        # Default geometric center (floating point)
        center_coords = np.array([D / 2 - 0.5, H / 2 - 0.5, W / 2 - 0.5])
    else:
        # User-specified center
        cx, cy, cz = center
        center_coords = np.array([cz, cy, cx])
    offset_z = np.random.randint(-radius, radius + 1)
    offset_y = np.random.randint(-radius, radius + 1)
    offset_x = np.random.randint(-radius, radius + 1)

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
    print(center, center_coords)
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
        p_z = center_coords[0] + offset_z + grid_y * v_vec[0] + grid_x * u_vec[0]
        p_y = center_coords[1] + offset_y + grid_y * v_vec[1] + grid_x * u_vec[1]
        p_x = center_coords[2] + offset_x + grid_y * v_vec[2] + grid_x * u_vec[2]

        coords = np.stack([p_z, p_y, p_x])

        # Interpolated sampling
        # mode='constant', cval=0.0 handles the out-of-bounds case
        patch = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
        slices.append(patch)

    patches = np.stack(slices, axis=0)
    return patches.astype(np.float32)

def crop_24channels_x(img, img_size=24):
    crop_volume = crop_24channels(img, img_size)
    crop_volume = crop_volume.transpose(2, 0, 1).astype(np.float32) # (x, z, y)
    return crop_volume

def crop_24channels_y(img, img_size=24):
    crop_volume = crop_24channels(img, img_size)
    crop_volume = crop_volume.transpose(1, 0, 2).astype(np.float32) # (y, z, x)
    return crop_volume

def crop_24channels_z(img, img_size=24):
    crop_volume = crop_24channels(img, img_size)
    crop_volume = crop_volume.astype(np.float32) #  (z, y, x)
    return crop_volume

def crop_24channels(img, img_size):
    d, h, w = img.shape
    z_floor, z_ceil = img_size // 2, img_size - (img_size // 2)
    y_floor, y_ceil = img_size // 2, img_size - (img_size // 2)
    x_floor, x_ceil = img_size // 2, img_size - (img_size // 2)

    z_min, z_max = z_floor, d - z_ceil
    y_min, y_max = y_floor, h - y_ceil
    x_min, x_max = x_floor, w - x_ceil

    valid = False
    crop_volume = None
    while not valid:
        cz = np.random.randint(z_min, z_max+1)
        cy = np.random.randint(y_min, y_max+1)
        cx = np.random.randint(x_min, x_max+1)

        crop_volume = img[cz-z_floor:cz+z_ceil, cy-y_floor:cy+y_ceil, cx-x_floor:cx+x_ceil]

        if is_valid_data(crop_volume):
            valid = True
    return crop_volume

def crop_3D(img, img_size=24):
    d, h, w = img.shape
    z_floor, z_ceil = img_size // 2, img_size - (img_size // 2)
    y_floor, y_ceil = img_size // 2, img_size - (img_size // 2)
    x_floor, x_ceil = img_size // 2, img_size - (img_size // 2)

    z_min, z_max = z_floor, d - z_ceil
    y_min, y_max = y_floor, h - y_ceil
    x_min, x_max = x_floor, w - x_ceil

    valid = False
    crop_volume = None
    while not valid:
        cz = np.random.randint(z_min, z_max+1)
        cy = np.random.randint(y_min, y_max+1)
        cx = np.random.randint(x_min, x_max+1)

        crop_volume = img[cz-z_floor:cz+z_ceil, cy-y_floor:cy+y_ceil, cx-x_floor:cx+x_ceil]

        if is_valid_data(crop_volume):
            valid = True
    return crop_volume[np.newaxis, ...].astype(np.float32)


def crop_3D_new(img, center, radius, img_size):
    d, h, w = img.shape
    cx, cy, cz = [int(np.round(coord)) for coord in center]
    floor = img_size // 2
    ceil = img_size - floor
    offset_z = np.random.randint(-radius, radius + 1)
    offset_y = np.random.randint(-radius, radius + 1)
    offset_x = np.random.randint(-radius, radius + 1)

    new_cz = cz + offset_z
    new_cy = cy + offset_y
    new_cx = cx + offset_x

    # Boundary check and padding
    z0, z1 = new_cz - floor, new_cz + ceil
    y0, y1 = new_cy - floor, new_cy + ceil
    x0, x1 = new_cx - floor, new_cx + ceil

    pad_z0, pad_z1 = max(0, -z0), max(0, z1 - d)
    pad_y0, pad_y1 = max(0, -y0), max(0, y1 - h)
    pad_x0, pad_x1 = max(0, -x0), max(0, x1 - w)

    crop = img[max(0, z0):min(d, z1),
               max(0, y0):min(h, y1),
               max(0, x0):min(w, x1)]

    if any((pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1)):
        padded = np.zeros((img_size, img_size, img_size), dtype=np.float32)
        padded[pad_z0:pad_z0 + crop.shape[0],
               pad_y0:pad_y0 + crop.shape[1],
               pad_x0:pad_x0 + crop.shape[2]] = crop
        crop = padded

    return crop[np.newaxis, ...].astype(np.float32)


def crop_near_sample(img, center, radius=4, img_size=24):
    d, h, w = img.shape
    floor, ceil = img_size // 2, img_size - (img_size // 2)
    cx, cy, cz = [int(np.round(coord)) for coord in center]
    offset_z = np.random.randint(-radius, radius + 1)
    offset_y = np.random.randint(-radius, radius + 1)
    offset_x = np.random.randint(-radius, radius + 1)
    new_cz = cz + offset_z
    new_cy = cy + offset_y
    new_cx = cx + offset_x

    crop_volume = img[new_cz - floor:new_cz + ceil,
                  new_cy - floor:new_cy + ceil,
                  new_cx - floor:new_cx + ceil]

    return crop_volume.astype(np.float32)


def crop_24channels_x_new(img, center, radius=4, img_size=24):
    crop_volume = crop_near_sample(img, center, radius, img_size)
    crop_volume = crop_volume.transpose(2, 0, 1).astype(np.float32) # (x, z, y)
    return crop_volume

def crop_24channels_y_new(img, center, radius=4, img_size=24):
    crop_volume = crop_near_sample(img, center, radius, img_size)
    crop_volume = crop_volume.transpose(1, 0, 2).astype(np.float32) # (y, z, x)
    return crop_volume

def crop_24channels_z_new(img, center, radius=4, img_size=24):
    crop_volume = crop_near_sample(img, center, radius, img_size)
    crop_volume = crop_volume.astype(np.float32) #  (z, y, x)
    return crop_volume


def is_valid_data(crop_volume):
    if np.min(crop_volume) > 0:
        return True
    return False



def get_crop_function(method, in_channels, channels_24_direction=None):
    if method == '2d':
        if in_channels == 3:
            return crop_3channels
        elif in_channels == 9:
            return crop_9channels
        else:
            if channels_24_direction is None:
                channels_24_direction = 'z'

            if channels_24_direction == 'x':
                return crop_24channels_x
            elif channels_24_direction == 'y':
                return crop_24channels_y
            else:
                return crop_24channels_z
    else:
        return crop_3D

def get_crop_function_new(method, in_channels, channels_24_direction=None):
    if method == '2d':
        if in_channels == 3:
            return crop_3channels
        elif in_channels == 9:
            return crop_9channels
        else:
            if channels_24_direction is None:
                channels_24_direction = 'z'

            if channels_24_direction == 'x':
                return crop_24channels_x_new
            elif channels_24_direction == 'y':
                return crop_24channels_y_new
            else:
                return crop_24channels_z_new
    else:
        return crop_3D_new