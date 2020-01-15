"""
2차원 convoluton(합성곱) 연산
"""
import numpy as np


def convolution_2d(x, w):
    """
    x, w: 2d ndarray, x.shape >= w.shape
    x와 w의 cross_correlation 결과 리턴
    convolution 결과 행렬의 shape :
    """
    conv = []
    xh, xw = x.shape[0], x.shape[1]
    wh, ww = w.shape[0], w.shape[1]
    result_row = xh - wh + 1
    result_col = xw - ww + 1

    for i in range(result_row):
        for j in range(result_col):
            x_sub = x[i:wh+i, j:ww+j]
            fma = np.sum(x_sub * w)
            conv.append(fma)
    conv = np.array(conv)
    return conv.reshape((result_row, result_col))


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 10).reshape((3,3))
    print('x =', x)
    w = np.array([[2, 0], [0, 0]])

    # 2d 배열 x의 가로(xw), 세로(xh)
    xh, xw = x.shape[0], x.shape[1]
    wh, ww = w.shape[0], w.shape[1]

    x_sub1 = x[0:wh, 0:ww]
    print(x_sub1)
    fma1 = np.sum(x_sub1 * w)
    print(fma1)

    x_sub2 = x[0:wh, 1:1+ww]
    fma2 = np.sum(x_sub2 * w)
    print(fma2)

    x_sub3 = x[1:1+wh, 0:ww]
    fma3 = np.sum(x_sub3 * w)
    print(fma3)

    x_sub4 = x[1:1+wh, 1:1+ww]
    fma4 = np.sum(x_sub4 * w)
    print(fma4)

    print('convolution_2d\n', convolution_2d(x, w))
    print()
    
    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    print(x)
    print(w)
    print('convolution_2d\n', convolution_2d(x, w))

