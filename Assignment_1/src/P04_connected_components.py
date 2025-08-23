import numpy as np
import skimage
from src.P01_histogram import *
from src.P02_otsu_binarization import *
from src.P03_adaptive_binarization import *




def connected_components(img):

    image = otsu(img)
    image = ~image

    con_img = np.zeros_like(image, dtype=np.uint8)
    num_components = 0
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            neighbours = np.array([image[i-1, j-1], image[i-1, j], image[i-1, j+1], image[i, j-1]])
            con_neighbours = np.array([con_img[i-1, j-1], con_img[i-1, j], con_img[i-1, j+1], con_img[i, j-1]])
            active_labels = con_neighbours[neighbours == 1]
                
            if image[i, j] == 1:
                if np.all(neighbours == 0):
                    num_components += 1
                    con_img[i, j] = num_components
                else:
                    if np.unique(active_labels).size == 1:
                        con_img[i, j] = active_labels[0]
                    else:
                        con_img[i, j] = np.min(active_labels)
                        mask = np.isin(con_img, active_labels)
                        con_img[mask] = con_img[i, j]
    
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    hist_con = histogram(con_img)
    hist_con[0] = -1
    k = np.argmax(hist_con)

    output_image[:,:,:][con_img > 0] = 0
    output_image[:,:,:][con_img == 0] = 255
    output_image[:,:,0][con_img == k] = 255

    return output_image