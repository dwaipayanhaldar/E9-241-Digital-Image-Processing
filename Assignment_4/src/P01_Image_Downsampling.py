import numpy as np
from scipy.ndimage import convolve

def downsample(image, N):
    return image[::N, ::N]

def gaussian_filter(k, sigma):
    x, y = np.meshgrid(np.arange(-(k//2),k//2+1),np.arange(-(k//2),k//2+1))
    G_xy = np.exp(-(x**2+y**2)/(2*sigma**2))
    G_xy /= np.sum(G_xy)
    return G_xy

def aa_downsample(image, window, sigma, N):
    gf = gaussian_filter(window, sigma)
    convolved_image = convolve(image, gf)
    return convolved_image[::N,::N]

def mse(img_ref, img):
    return np.mean(np.abs(img - img_ref) ** 2)

