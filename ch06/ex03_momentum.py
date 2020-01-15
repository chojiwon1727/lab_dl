"""
파라미터 최적화 알고리즘
2) Momentum 알고리즘
    - v : 속도(velocity)
    - m : 모멘텀 상수(momentum constant)
    - lr : 학습률(learning rate)
    - w : 파라미터
        -> v = m * v - lr * dl/dw
        -> W = W + v  = W + m *v - lr * dl/dw
        -> sgd에 +m*v를 해준 것
"""
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01_matplot3d import fn, fn_derivative


class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr        # 학습률
        self.m = m          # 모멘텀상수(속도 v에 곱해줄 상수)
        self.v = dict()     # 속도(각 파라미터(x, y) 방향의 속도를 저장하기 위한 dict)

    def update(self, params, gradients):
        if not self.v:   # v dict에 원소가 없으면
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape의 0으로 채워진 배열 생성
                self.v[key] = np.zeros_like(params[key])

        # 속도 v, 파라미터 params를 갱신(update)하는 기능
        if self.v:
            for key in params:
                self.v[key] = self.v[key] * self.m - self.lr * gradients[key]
                params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 인스턴스 생성
    momentum = Momentum(lr=0.05)
    # update 메소드 테스트
    init_position = (-7, 2)
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]
    # params = {'x':-7., 'y':2.} 해도 됨

    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0

    x_hist = []
    y_hist = []
    epoch = 30
    for i in range(epoch):
        x_hist.append(params['x'])
        y_hist.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    for x, y in zip(x_hist, y_hist):
        print(f'({x}, {y})')

    # 등고선 그래프에 파라미터 변화 그래프 추가
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    plt.contour(X, Y, Z, 10)
    plt.title('Momentum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_hist, y_hist, 'o-', color='red')
    plt.show()

