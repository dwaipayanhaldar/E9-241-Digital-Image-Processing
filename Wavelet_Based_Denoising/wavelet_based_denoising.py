import numpy as np
from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec

def mad(x):
    return 1.482579 * np.median(np.abs(x - np.median(x)))

def Visu_Shrink(image):
    m = image.shape[0]
    thr = mad(image) * np.sqrt(2*np.log(m))
    return thr