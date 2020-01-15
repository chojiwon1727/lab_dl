"""
1차원 Convolution, Cross-Correlation 연산
"""
import numpy as np


def convolution_1d(x, w):
    """ x, w : 1d ndarray, len(x) >= len(w) """
    w_r = np.flip(w)
    conv = []
    len_result = len(x) - len(w) + 1
    for i in range(len_result):
        x_sub = x[i:i+len(w)]
        fma = np.sum(x_sub * w_r)
        conv.append(fma)
    conv = np.array(conv)
    return conv


def cross_correlation_1d(x, w, convolution=False):
    """ x, w : 1d ndarray, len(x) >= len(w) """
    if convolution == True:
        w = np.flip(w)

    conv = []
    len_result = len(x) - len(w) + 1
    for i in range(len_result):
        x_sub = x[i:i + len(w)]
        fma = np.sum(x_sub * w)
        conv.append(fma)
    conv = np.array(conv)
    return conv


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    w = np.array([2, 1])
    w_r = np.flip(w)

    conv = []
    for i in range(4):
        x_sub = x[i:i+len(w)]   # (0,1), (1,2), (2,3), (3,4)
        fma = np.sum(x_sub * w_r) # 1차원인 경우, np.dot(x_sub, w_r) 동일
        conv.append(fma)
    conv = np.array(conv)
    print('conv =', conv)
    print('conv =', convolution_1d(x, w))

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    print('conv =', convolution_1d(x, w))

    # 교차상관(Cross-correlation)
    # 합성곱 연산과 다른 점은 w를 반전시키지 않는 것
    # CNN(Convolutional Neural Network, 합성곱 신경망)에서는 대부분 교차상관 사용

    cross_correlation = cross_correlation_1d(x, w)
    print('cross_correlation =', cross_correlation)