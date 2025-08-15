import skimage.io

img = skimage.io.imread('images/coins.png')

print(img.size)
print(img.ravel().shape)