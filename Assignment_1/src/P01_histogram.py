import numpy as np
import skimage
import matplotlib.pyplot as plt


def meanintensity(image, hist):
    cal_avg_intensity = 0
    for pixel_value in range(hist.size):
        hist_pmf = hist[pixel_value] / image.size
        cal_avg_intensity += (pixel_value * hist_pmf)
    return cal_avg_intensity

def histogram(image):
    frequency_list = np.zeros(256, dtype=int)
    for pixel_value in image.ravel():
        frequency_list[pixel_value] += 1
    return frequency_list



if __name__ == "__main__":
    img = skimage.io.imread('images/coins.png')
    hist = histogram(img)
    # Comparing with the library function
    hist_lib, _ = np.histogram(img, bins=256, range=(0, 255))


    print("Average Intensity:", meanintensity(img, hist))
    print("Average Intensity (Library):", np.mean(img))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Custom Histogram')
    plt.bar(range(256), hist, color='blue', alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.title('Library Histogram')
    plt.bar(range(256), hist_lib, color='red', alpha=0.7)
    plt.show()