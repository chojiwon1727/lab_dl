import numpy as np


def numerical_diff(fn, x):
    """
    numerical_differential
    함수 fn과 점 x가 주어졌을 때 x에서의 함수 fn의 미분 값(도함수 값)
    """
    h = 1e-04  # 0.0001 -> 너무 작으면 under flore 발생
    return (fn(x+h) - fn(x-h)) / (2*h)


def f1(x):
    return 0.001 * x**2 + 0.01 * x


def f1_prime(x):
    """ 근사값을 사용하지 않은 함수 f1의 도함수 """
    return 0.002 * x + 0.01


def f2(x):
    """ x =[x0, x1, ...] """
    return np.sum(x**2)   # x0**2 + x1**2 + ...


def _numerical_gradient(fn, x):
    """
    점 x = [x0, x1, x2, ..., xn]에서의 함수 fn(x0, x1, x2, ..., xn)의
    각 편미분(partial differential)값들의 배열을 리턴

    :param fn: 독립변수를 여러개 갖는 함수
    :param x: n차원 array,
    :return: 편미분 값들의 array
    """
    x = x.astype(np.float, copy=False)   # 실수 타입
    gradient = np.zeros_like(x)   # np.zeros(shape = x.shape)과 동일
    h = 1e-04  #0.0001
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)                 # x[i]가 아닌 다른 원소들은 상수 취급 -> 상수 미분해봤자 0
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2*h)
        x[i] = ith_value
    return gradient


def numerical_gradient(fn, x):
    """
    x = [[], [], ...] -> 2차원 배열
    """
    if x.ndim == 1:
        return _numerical_gradient(fn, x)
    else:
        grads = np.zeros_like(x)
        for i, x_i in enumerate(x):
            grads[i] = _numerical_gradient(fn, x_i)
        return grads



# def f3(x):
#     for i in range(x.size):
#         X.append(x**i+1)
#     return np.array(X)

# def  f4(x):
#     x = x.


if __name__ == '__main__':
    # f1 함수를 사용했을 경우
    estimate = numerical_diff(f1, 3)
    print('근사값:', estimate)    # 0.016000000000043757
    real = f1_prime(3)
    print('실제값:', real)        # 0.016

    # f2 함수를 사용했을 경우  ->  점 (3,4)에서의 편미분(변수 1개를 상수로 취급) 값
    estimate1 = numerical_diff(lambda x: x**2 + 4**2, 3)   # 3에서의 편미분 -> 4는 상수 취급하므로
    print(estimate1)
    estimate2 = numerical_diff(lambda x: x**2 + 3**2, 4)
    print(estimate2)

    gradient = numerical_gradient(f2, np.array([3,4]))
    print(gradient)

    # f3 = x0 + x1**2 + x2**3 함수에서 점 (1,1,1)에서의 각 편미분들의 값
    # df/dx0 = 1, df/dx1 = 2, df/dx2 = 3
    # gradient2 = numerical_gradient(f3, np.array([1,1,1]))
    # print(gradient2)
    #
    # print()
    # a = np.array([1,1,1])
    # print(a)
    # print(a[0])
    # print(a.size)

    # f4 = x0**2 + 2*x0*x1 + x1**2 함수에서 점(1,2)에서의 각 편미분들의 값
    # df/dx0 = 4, df/dx1 = 5