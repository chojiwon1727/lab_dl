import numpy as np
from scipy.signal import convolve, correlate, convolve2d, correlate2d

from ch07.ex01_convolution1d import convolution_1d

if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print(x)
    w = np.array([2, 0])
    print(convolution_1d(x, w))

    # x의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print('pad_width = 1 :', convolution_1d(x_pad, w))

    # convolution 결과의 크기가 입력데이터 x와 동일한 크기가 되도록 padding
    x_pad = np.pad(x, pad_width=(1, 0), mode='constant', constant_values=0)
    print('pad_width = (1,0) :',convolution_1d(x_pad, w))

    x_pad = np.pad(x, pad_width=(0, 1), mode='constant', constant_values=0)
    print('pad_width = (0,1) :',convolution_1d(x_pad, w))

    conv = convolve(x, w, mode='valid')
    print('mode=valid', conv)

    conv = convolve(x, w, mode='full')
    print('mode=full', conv)

    conv = convolve(x, w, mode='same')
    print('mode=same', conv)

    cross_corr = correlate(x, w, mode='full')

    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])

    w = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])

    conv = correlate2d(x, w, mode='valid')
    print('mode valid\n', conv)
    conv = correlate2d(x, w, mode='full')
    print('mode full\n', conv)
    conv = correlate2d(x, w, mode='same')
    print('mode same\n', conv)


