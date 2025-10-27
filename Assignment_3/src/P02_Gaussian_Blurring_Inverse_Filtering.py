import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter_td(k = 13, sigma = 2.5):
    x, y = np.meshgrid(np.arange(-(k//2),k//2+1),np.arange(-(k//2),k//2+1))
    G_xy = np.exp(-(x**2+y**2)/(2*sigma**2))
    G_xy /= np.sum(G_xy)
    return G_xy

def gaussian_filter_fd(k,M,N):
    i,j = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    i = i -(M-1)/2
    j = j - (N-1)/2
    H_cont =  np.exp(-k*(i**2+j**2))
    return H_cont

if __name__ == "__main__":
    G = gaussian_filter_td()
    G_kernel_padded = np.pad(G, ((6,6),(6,6)))
    plt.imshow(G_kernel_padded, cmap= 'gray')
    plt.show()