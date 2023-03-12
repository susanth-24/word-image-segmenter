import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
import cv2

image = cv2.imread("./word_segmenter/image2.jpg")

#function that returns binary image!
def binary(image):
    #converting the image to gray to remove noice
    grey_image=skimage.color.rgb2gray(image)
    blur_image=skimage.filters.gaussian(grey_image,sigma=1.0)
    #getting the threshold value
    threshold=skimage.filters.threshold_otsu(blur_image)
    binary_image=blur_image<threshold
    #returning the binary black and white image
    binary_image = binary_image.astype(np.uint8) * 255
    print(binary_image)
    return binary_image

""" p=binary(image)
plt.imshow(p,cmap="gray")
plt.show() """