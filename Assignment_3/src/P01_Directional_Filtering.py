import numpy as np

def vertical_sin(M):
    image = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            image[i,j] = np.sin((2*np.pi*12*i)/M)
    
    return image

def horizontal_sin(M):
    image = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            image[i,j] = np.sin((2*np.pi*8*j)/M)
    
    return image

def diagonal_sin(M):
    image = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            image[i,j] = np.sin((2*np.pi*(6*i+10*j))/M)
    
    return image 

def directional_filter(dft_img, theta_min, theta_max):
    M, N = dft_img.shape
    H_i, H_j = np.meshgrid(np.arange(M), np.arange(N))
    H_i -= (M//2)
    H_j -= (N//2)
    theta = 1*np.arctan2(H_j,H_i)

    theta_min_radian = (theta_min*np.pi)/180
    theta_max_radian = (theta_max*np.pi)/180

    H = np.where((theta >= theta_min_radian) & (theta<= theta_max_radian), 1,0)
    return H

def mse(img_ref,img):
    M, N = img_ref.shape
    diff_sq = np.abs((img - img_ref)**2)
    mse = np.sum(diff_sq)/(M*N)
    return mse

    
