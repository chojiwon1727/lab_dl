"""
weight 행렬에 경사 하강법(gradient descent) 적용
"""
import numpy as np
from ch03.ex11 import softmax
from ch04.ex03 import cross_entropy
from ch04.ex05 import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2,3)    # randn : 정규분포(normal distribution)를 따르는 랜덤 넘버 생성
                                         # -> 가중치 행렬의 초기값들을 임의로 설정

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, y_true):
        """
        손실함수(loss function) - 분류문제(cross entropy)
        """
        y_pred = self.predict(x)
        ce = cross_entropy(y_pred, y_true)
        return ce

    def gradient(self, x, t):
        """ W 행렬에 대한 손실함수의 기울기 """
        fn = lambda W: self.loss(x, t)
        return numerical_gradient(fn, self.W)


if __name__ == '__main__':
    network = SimpleNetwork()
    print('W:', network.W)

    # x = [0.6, 0.9]일 때, y_true = [0, 0, 1]이라고 가정
    x = np.array([0.6, 0.9])
    y_true = np.array([0.0, 0.0, 1.0])
    print('x :', x)
    print('y_true :', y_true)

    # y_pred = network.predict(x)
    # print('y_pred :', y_pred)
    #
    # ce = network.loss(x, y_true)
    # print('cross entropy :', ce)
    #
    # # y_pred값이 y_true값에 근사할 수 있도록 만드는 W 행렬의 값을 찾아야 함
    # # -> y_pred와 y_true값의 오차를 이용한 cross entropy를 줄여나가는 gradient를 계산하면 됨
    #
    # g1 = network.gradient(x, y_true)
    # print('g1 :', g1)               # 결과: (2,3)행렬, 각 W값의 변화율
    # print()
    #
    # lr = 0.1
    # network.W -= lr * g1
    # print('W :', network.W)
    # print('y_pred:', network.predict(x))
    # print('ce =', network.loss(x, y_true))

    # 55 ~ 63번째 라인을 for문안에서 100번 반복
    lr = 0.6
    step = 100
    for i in range(step):
        gradient = network.gradient(x, y_true)
        network.W -= lr * gradient
        print(f'\n>>> {i+1}번째 시행')
        print('gradient :\n', gradient)
        print('W :\n', network.W)
        print('y_pred :', network.predict(x))
        print('cross entropy :', network.loss(x, y_true))
        print(f'max probability : {network.predict(x).argmax()+1}번째')
