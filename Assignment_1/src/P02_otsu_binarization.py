import numpy as np
import skimage
from src.P01_histogram import histogram

# def class_probability(hist, t):
#     hist_norm = hist/np.sum(hist)
#     omega_0 = np.sum(hist_norm[:t+1])
#     return omega_0, 1-omega_0

# def class_mean(hist, t):
#     omega_0, omega_1 = class_probability(hist, t)
#     hist_norm = hist/np.sum(hist)
#     mu_0 = (np.sum(np.arange(t+1) * hist_norm[:t+1]))/omega_0 if omega_0 > 0 else 0
#     mu_1 = (np.sum(np.arange(t+1, 256) * hist_norm[t+1:]))/omega_1 if omega_1 > 0 else 0
#     return mu_0, mu_1

# def class_variance(hist, t):
#     omega_0, omega_1 = class_probability(hist, t)
#     mu_0, mu_1 = class_mean(hist, t)
#     hist_norm = hist/np.sum(hist)
#     sigma_0_sq= (np.sum(((np.arange(t+1) - mu_0) ** 2) * hist_norm[:t+1]))/ omega_0 if omega_0 > 0 else 0
#     sigma_1_sq = (np.sum(((np.arange(t+1, 256) - mu_1) ** 2) * hist_norm[t+1:])) / omega_1 if omega_1 > 0 else 0
#     return sigma_0_sq, sigma_1_sq


def within_class_variance(image, t):

    # omega_0, omega_1 = class_probability(hist, t)
    # sigma_0_sq, sigma_1_sq = class_variance(hist, t)

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

    return sigma_w_squared[t]


def between_class_variance(image, t):
    # hist = histogram(image)
    # omega_0, omega_1 = class_probability(hist, t)
    # mu_0, mu_1 = class_mean(hist, t)

    hist = histogram(image)  
    hist_norm = hist/np.sum(hist)

    omega = np.cumsum(hist_norm)                  
    mu_0 = np.cumsum(hist_norm * np.arange(256))  
    mu_t = mu_0[-1]    

    mu_1 = mu_0 / (omega + 1e-12)                       
    mu_2 = (mu_t - mu_0) / (1 - omega + 1e-12) 

    sigma_b_squared = omega*(1-omega)*(mu_1 - mu_2)**2

    return sigma_b_squared[t]
