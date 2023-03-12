import numpy as np
from binary import binary
from segmenter import segmenter
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters

#add any image
img=cv2.imread("./word_segmenter/image2.jpg")

word_images=segmenter(img)
plt.imshow(word_images[51])
plt.show()