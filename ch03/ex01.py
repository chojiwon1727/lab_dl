"""
Perceptron:
    -입력 : (x1, x2)
    -출력 :
        a = x1*w1 + x2*w2 + b 계산 (w :가중치, b:bias)
        y = 1, a > 임계값
          = 0, a <= 임계값      -> a값이 크다, 작다를 결정하는 것이 활성화 함수
    (분류문제면 1, 0이 나오고, 회귀이면 연속된 결과를 줄 것)

신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해 주는 함수가 존재
-> 활성화 함수(activation function)
"""
import numpy as np
import math
import matplotlib.pyplot as plt


# 활성화 함수1 :  계단 함수
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function2(x):
    """
    Step Function

    :param x: numpy.ndarray
    :return: step 함수 출력(0,1)로 이루어진 numpy.ndarray
    """
    # result = []
    # for x_i in x:
    #     if x_i > 0:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return np.array(result)

    y = x > 0   # np.ndarray 자체가 원소간 비교를 하기 때문에 이 자체에 for문이 들어간 것과 같음(F, T의 array) -> F, T를 0,1로만 바꾸면 됨
    return y.astype(np.int)   # ndarray.astype(np.int)   -> array의 다른 타입을 int타입으로 변환해주는 함수


# 활성화 함수2 : 시그모이드 함수
def sigmoid(x):
    """
    sigmoid = 1 / (1 + exp(-x))

    * exp *
    math.exp : x가 number만 가능
    np.exp : x가 number, ndarray, iterable 다 가능
    """
    return 1 / (1 + np.exp(-x))


# 활성화 함수3
def relu(x):
    """
    RuLU(Rectified Linear Unit)
     y = x, if x > 0
       = 0, otherwise(x < 0이면 0으로 표현)
    """
    # result = []
    # for x_i in x:
    #     if x_i > 0:
    #         result.append(x_i)
    #     else:
    #         result.append(0)
    # return np.array(result)

    # result = [x_i if x_i > 0 else 0 for x_i in x]
    # return np.array(result)

    return np.maximum(0, x)





if __name__ == '__main__':
    x = np.arange(-3, 4)     # 파이썬의 range()함수와 유사(시작값부터 끝나는 값까지 1씩 증가시키는 함수)
    print('x = ', x)

    # step 함수
    for x_i in x:                     # array의 값을 넘기는 것
        print(step_function(x_i), end='')
    print()

    print('y = ', step_function2(x))     # array의 값을 넘기는 것이 아니라 array 자체를 넘기면 array를 리턴해주면 좋겠음

    # sigmoid 함수
    print('sigmoid : ', sigmoid(x))

    x_points = np.linspace(-10, 10, 100)
    step = step_function2(x_points)
    sigmoid = sigmoid(x_points)

    plt.plot(x_points, step, label='step function')
    plt.plot(x_points, sigmoid, label='sigmoid function')
    plt.legend()
    plt.show()

    # relu 함수
    relu = relu(x_points)
    print(relu)
    plt.plot(x_points, relu, label='relu function')
    plt.legend()
    plt.show()
