import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def laplacian_of_gaussian(image, k, sigma):
    x, y = np.meshgrid(np.arange(-(k//2), k//2 + 1), np.arange(-(k//2), k//2 + 1))
    norm = (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
    LoG = norm * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    LoG -= LoG.mean()
    return convolve(image, LoG)

def prewitt(image):
    prewitt_kernel_x = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
    prewitt_kernel_y = np.array([[-1,-1,-1], [0,0,0],[1,1,1]])
    I_x = convolve(image, prewitt_kernel_x)
    I_y = convolve(image, prewitt_kernel_y)
    return np.sqrt(I_x**2+I_y**2)

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
