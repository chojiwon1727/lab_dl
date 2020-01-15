"""
CNN(Convolutional Neural Network, 합성곱 신경망)
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve, correlate

img = Image.open('sample.jpg')
img_pixel = np.array(img)
print(img_pixel.shape)

plt.imshow(img_pixel)
plt.show()

# image RED 정보
print(img_pixel[:, :, 0])

# (3, 3, 3) 필터
filter = np.zeros((3, 3, 3))
filter[1, 1, 0] = 1.0
transformed = convolve(img_pixel, filter, mode='same') / 255
plt.imshow(transformed)
plt.show()