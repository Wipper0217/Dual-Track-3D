import math

import numpy as np


def get_cross_line(img, point, wh):
    row = img[int(point[1]), int(point[0]-wh[0]/2):int(point[0]+wh[0]/2), 0]
    col = img[int(point[1]-wh[1]/2):int(point[1]+wh[1]/2), int(point[0]), 0]
    a = int(wh[1]) if wh[0]>wh[1] else int(wh[0])

    square = img[int(point[1])-a//2:int(point[1])+a//2, int(point[0])-a//2:int(point[0])+a//2, 0]
    diag1 = square.diagonal()
    diag2 = np.fliplr(square).diagonal()

    return np.hstack((row, col, diag1, diag2))


def get_v_threshold(points):
    points_later = np.vstack((points[0,:], points[0:-1, :]))
    v = points - points_later
    v_value = np.sqrt(np.square(v[:, 0]) + np.square(v[:, 1]))
    v_m = np.mean(v_value[:100])
    v_s = np.std(v_value[:100])
    return v_m+7 * v_s


def keep_origin(origin, fix, init, weighted_o, weighted_f):
    origin = (origin - np.mean(origin)) / (np.std(origin) * len(origin))
    fix = (fix - np.mean(fix)) / (np.std(fix) * len(fix))

    origin_p = np.correlate(origin, init)[0]
    fix_p = np.correlate(fix, init)[0]
    # print(f'origin_p*weighted_o: {origin_p:.2f}*{math.exp(weighted_o):.2f}={origin_p*math.exp(weighted_o):.3f}\n
    # fix_p*weighted_f: {fix_p:.2f}*{math.exp(weighted_f):.2f}={fix_p*math.exp(weighted_f):.3f}')
    return origin_p*math.exp(weighted_o) >= fix_p*math.exp(weighted_f)
    # return origin_p >= fix_p

def drift_correction(img, initial_size, tcv, predict_boxes, current_box, current_idx, v_threshold=0):
    """

    Args:
        img: current frame image
        predict_boxes: previous predicted boxes
        current_box: current predicted box, shape=[xmin, ymin, width, height]
        current_idx: current frame index
        v_threshold:

    Returns:

    """
    vf = current_box -predict_boxes[current_idx-1, :]
    if np.square(vf[0])+np.square(vf[1]) > np.square(v_threshold):
        v_weighted = (v_threshold - np.sqrt(np.square(vf[0]) + np.square(vf[1]))) / 2 / v_threshold
        v_fix = predict_boxes[current_idx - 1, :] - predict_boxes[current_idx - 2, :]
        fix_predict = predict_boxes[current_idx - 1, :] + v_fix
        v_fix_weighted = (v_threshold - np.sqrt(np.square(v_fix[0]) + np.square(v_fix[1]))) / 2 / v_threshold
        point_origin = current_box[:2] + (current_box[2:] - 1) / 2
        origin = get_cross_line(img, point_origin, (initial_size[0], initial_size[1]))
        point_fix = fix_predict[:2] + (fix_predict[2:] - 1) / 2
        # print(f, point_fix, point_origin)
        if (point_fix[0] < 0) or (point_fix[0] > img.shape[0]) or \
                (point_fix[1] < 0) or (point_fix[1] > img.shape[1]):
            point_fix = predict_boxes[current_idx - 1, :]
            point_fix = point_fix[:2] + (point_fix[2:] - 1) / 2
        fix = get_cross_line(img, point_fix, (initial_size[0], initial_size[1]))

