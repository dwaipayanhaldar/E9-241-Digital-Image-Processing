import numpy as np

def bil_int(image, a_1, a_2):
    

    m = int(np.floor(a_1))
    n = int(np.floor(a_2))

    b_1 = (a_1-m)*(a_2-n)
    b_2 = (a_1-m)*(n+1-a_2)
    b_3 = (m+1-a_1)*(a_2-n)
    b_4 = (m+1-a_1)*(n+1-a_2)

    return image[m,n]*b_4 + image[m,n+1]*b_3 + image[m+1,n]*b_2 + image[m+1,n+1]*b_1

def upscale(image, b = 2):
    H, W = image.shape
    upscaled_image = np.zeros((b*H, b*W))

    for i in range(1, upscaled_image.shape[0] - 2):
        for j in range(1, upscaled_image.shape[1] - 2):
            upscaled_image[i,j] = bil_int(image, i/2,j/2)
    
    return upscaled_image

def rotate_image(img, theta):
    H, W = img.shape
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    corners = np.array([[0, 0], [H, 0], [0, W], [H, W]])

    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    rotated_corners = corners @ R.T

    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)

    new_H = int(np.floor(x_max - x_min))
    new_W = int(np.floor(y_max - y_min))

    rotated_img = np.zeros((new_H, new_W))
    shift_x = -x_min
    shift_y = -y_min

    for i in range(new_H):
        for j in range(new_W):
            xi = i- shift_x
            yj = j - shift_y

            x = cos_t * xi + sin_t * yj
            y = -sin_t * xi + cos_t * yj

            if 0 <= x < H-1 and 0 <= y < W-1:
                rotated_img[i, j] = bil_int(img, x, y)

    return rotated_img
