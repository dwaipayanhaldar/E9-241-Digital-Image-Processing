import numpy as np
from scipy.ndimage import convolve


def box_kernel(m):
    k = 1/(m*m)
    box = np.ones((m,m))*k
    return box


def sharpenAdjust(img, p):
    kernel = box_kernel(25)
    convolved_img = convolve(img, kernel)

    mask = img - convolved_img
    added_img = img + p * mask

    added_img_clipped = np.clip(added_img, 0, 255)

    return added_img_clipped
