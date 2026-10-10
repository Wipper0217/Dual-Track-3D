import numpy as np

def normalize(image):
    image = image.astype(np.float32)
    normalized_image = image / 255.0
    return normalized_image
