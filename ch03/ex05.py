import numpy as np

from ch03.ex01 import sigmoid


def init_network():
    """
    신경망(neural network)에서 사용되는 가중치(W) 행렬과 bias 행렬 생성
    입력값 : (x1, x2) -> 1x2 행렬
    은닉층 : 2개
            --> 1st 은닉층 : 뉴런 3개  (x @ w1 + b1)
            --> 2nd 은닉층 : 뉴런 2개 (z @ w2 + b2)
    출력층 : (y1, y2) -> 1x2 행렬
    w1, w2, w3, b1, b2, b3를 난수로 생성
    """
    np.random.seed(1224)
    network = dict()       # 가중치 / bias 행렬을 저장하기 위한 dict(key : w1, value: 난수) -> 최종 리턴값

    # 은닉층 1 : x @ w1 + b1
    # (1 x 2) @ (2 x 3) + (1 x 3)  =  (1 x 3)
    network['W1'] = np.random.random(size=(2,3)).round(2)     # -> size를 주면 0~1사이의 값을 size에 맞게 리턴
    network['b1'] = np.random.random(size=3).round(2)

    # 은닉층 2 : z1 @ w2 + b2
    # (1 x 3) @ (3 x 1) + (1 x 2)  =  (1 x 2)
    network['W2'] = np.random.random(size=(3, 2)).round(2)
    network['b2'] = np.random.random(size=2).round(2)

    # 출력층 : z2 @ W3 + b3
    # (1 x 2) @ (2 x 2) + (1 x 2)  =  (1 x 2)
    network['W3'] = np.random.random(size=(2, 2)).round(2)
    network['b3'] = np.random.random(size=2).round(2)

    return network


def forward(network, x):
    """
    순방향 전파(forward propagation) : 입력층 -> 은닉층 -> 출력층

    은닉층의 활성화 함수 : 시그모이드 함수

    :param network: 신경망에서 사용되는 가중치와 bias 행렬들을 저장한 dict
    :param x: 입력값을 가지고 있는 1차원 리스트  -> ex, [x1, x2]
    :return: 2개의 은닉층과 1개의 출력층을 거친 후 계산된 최종 출력값 -> ex, [y1, y2]
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1)+b1
    z1 = sigmoid(a1)

    z2 = sigmoid(z1.dot(W2)+b2)

    y = z2.dot(W3)+b3
    return softmax(y)


# 출력층의 활성화 함수 1 - 항등 함수 : 회귀문제
def identity_function(x):
    return x


# 출력층의 활성화 함수 2 - softmax : 분류문제
def softmax(x):
    """
    x = [x1, x2, x3, ..., x_k, ..., x_n]
    배열 x의 k번째 값의 softmax y_k = exp(x_k) / [sum k to n exp(x_k)] = exp(x_k) / sum(exp(x_k))
    --> 의미 : 배열 x에서 k가 차지하는 비율(확률) --> 0과 1사이의 값
    :param x: array
    :return: 0 ~ 1사이의 값(모든 리턴값의 총 합 : 1)
    """
    max_x = np.max(x)  # array x의 원소들 중 최대값을 찾음
    y = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return y


if __name__ == '__main__':
    network = init_network()

    x = np.array([1,2])
    y = forward(network, x)
    print('y :', y)

    # print('x =', x)
    # print('softmax =', softmax(x))
    #
    # x = [1,2,3]
    # print('softmax =', softmax(x))
    #
    # x = [1e0, 1e1, 1e2, 1e3]   # [1, 10, 100, 1000]
    # print('x =', x)
    print('softmax =', softmax(x))





