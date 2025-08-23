import numpy as np
import skimage
from src.P01_histogram import *
from src.P02_otsu_binarization import *
from skimage.util import view_as_windows


def otsu(image):
    
    hist = histogram(image)  
    hist_norm = hist/np.sum(hist)

    omega = np.cumsum(hist_norm)                  
    mu_0 = np.cumsum(hist_norm * np.arange(256))  
    mu_t = mu_0[-1]    

    mu_1 = mu_0 / (omega + 1e-9)                       
    mu_2 = (mu_t - mu_0) / (1 - omega + 1e-9) 

    sigma_b_squared = omega*(1-omega)*(mu_1 - mu_2)**2

    thresh = np.argmax(sigma_b_squared)

    return image > thresh



def adaptive_binarization(image, N):

    
    vote_count = np.zeros_like(image, dtype=int)
    total_count = np.zeros_like(image, dtype=int)

    step = int(N*0.8)
    for i in range(0,image.shape[0], step):
        for j in range(0, image.shape[1], step):
            patches = image[i:i+N, j:j+N]
            vote_count[i:i+N, j:j+N] += otsu(patches)
            total_count[i:i+N, j:j+N] += 1

    output_image = np.where(vote_count / total_count > 0.5, 255, 0)
            
            
    return output_image


