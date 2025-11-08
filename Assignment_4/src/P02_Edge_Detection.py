import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def histogram(image, bins=256):
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    return hist, bin_edges

def otsu(image, bins=256):
    hist, bin_edges = histogram(image, bins)
    hist_norm = hist / np.sum(hist)
    omega = np.cumsum(hist_norm)
    mu_0 = np.cumsum(hist_norm * np.arange(bins))
    mu_t = mu_0[-1]
    mu_1 = mu_0 / (omega + 1e-9)
    mu_2 = (mu_t - mu_0) / (1 - omega + 1e-9)
    sigma_b_squared = omega * (1 - omega) * (mu_1 - mu_2)**2
    best_thresh_index = np.argmax(sigma_b_squared)
    thresh = bin_edges[best_thresh_index]
    return thresh


def gaussian_filter_td(k = 13, sigma = 2.5):
    x, y = np.meshgrid(np.arange(-(k//2),k//2+1),np.arange(-(k//2),k//2+1))
    G_xy = np.exp(-(x**2+y**2)/(2*sigma**2))
    G_xy /= np.sum(G_xy)
    return G_xy

def zero_crossings(image, threshold):
    sign_image = np.sign(image)
    zero_cross_h = sign_image[:, :-1] * sign_image[:, 1:] < 0
    zero_cross_v = sign_image[:-1, :] * sign_image[1:, :] < 0
    zero_cross_d1 = sign_image[:-1, :-1] * sign_image[1:, 1:] < 0
    zero_cross_d2 = sign_image[:-1, 1:]  * sign_image[1:, :-1] < 0
    diff_h = np.abs(image[:, :-1] - image[:, 1:]) > threshold
    diff_v = np.abs(image[:-1, :] - image[1:, :]) > threshold
    diff_d1 = np.abs(image[:-1, :-1] - image[1:, 1:]) > threshold
    diff_d2 = np.abs(image[:-1, 1:] - image[1:, :-1]) > threshold
    zero_cross = np.zeros_like(image, dtype=np.uint8)
    zero_cross[:, 1:][zero_cross_h & diff_h] = 1
    zero_cross[1:, :][zero_cross_v & diff_v] = 1
    zero_cross[1:, 1:][zero_cross_d1 & diff_d1] = 1
    zero_cross[1:, :-1][zero_cross_d2 & diff_d2] = 1
    
    return zero_cross


def laplacian_of_gaussian(image, k, sigma, threshold= 0.25):
    x, y = np.meshgrid(np.arange(-(k//2), k//2 + 1), np.arange(-(k//2), k//2 + 1))
    norm = (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
    LoG = norm * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    LoG -= LoG.mean()
    convolved_image = convolve(image, LoG)

    return zero_crossings(convolved_image, threshold)

def prewitt(image):
    prewitt_kernel_x = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
    prewitt_kernel_y = np.array([[-1,-1,-1], [0,0,0],[1,1,1]])
    I_x = convolve(image, prewitt_kernel_x)
    I_y = convolve(image, prewitt_kernel_y)
    edge_image = np.sqrt(I_x**2+I_y**2)
    return edge_image > otsu(edge_image)

def plot_figure(original_img, img1, img2):
    plt.figure(figsize=(24,16))
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(original_img, cmap = 'gray')
    plt.subplot(1,3,2)
    plt.title("Edge Map(Prewitt)")
    plt.imshow(img1, cmap = 'gray')
    plt.subplot(1,3,3)
    plt.title("Edge Map(Laplacian of Gaussian)")
    plt.imshow(img2, cmap = 'gray')
    plt.show()

def add_gaussian_noise(image, std = 0.1):
    return np.random.normal(np.mean(image),std, (image.shape[0], image.shape[1])) + image

def gaussian_smoothing(image, window = 7, std = 3):
    return convolve(image, gaussian_filter_td(window, std))