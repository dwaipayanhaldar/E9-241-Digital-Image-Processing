import numpy as np


def histogram(image):
    frequency_list = np.zeros(256, dtype=int)
    for pixel_value in image.ravel():
        frequency_list[pixel_value] += 1
    return frequency_list

def otsu_within_class(image):
    hist = histogram(image)
    hist_norm = hist/np.sum(hist)
    i = np.arange(256)
    omega = np.cumsum(hist_norm)                 
    mu0 = np.cumsum(hist_norm * i)              
    nu0 = np.cumsum(hist_norm * (i**2))         

    muT = mu0[-1]
    nuT = nu0[-1]

    mu1 = mu0 / (omega + 1e-12)
    mu2 = (muT - mu0) / (1 - omega + 1e-12)

    sigma1_sq = (nu0 / (omega + 1e-12)) - mu1**2
    sigma2_sq = ((nuT - nu0) / (1 - omega + 1e-12)) - mu2**2

    sigma_w_squared = sigma1_sq * omega + sigma2_sq * (1 - omega)

    thresh = np.argmin(sigma_w_squared)

    return image> thresh, sigma_w_squared[thresh], thresh


def box_kernel(m):
    k = 1/(m*m)
    box = np.ones((m,m))*k
    return box

