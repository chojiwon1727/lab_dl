"""
신경망 학습 목적: 손실 함수의 값을 가능한 낮추는 파라미터(W, b)를 찾는 것
파라미터(parameter)
    - 파라미터 : 가중치(Weight), 편향(bias)
    - 하이퍼 파라미터(hyper parameter) : 학습률(learning rate), epoch, batch size, 신경망 은닉층 수, 뉴런 수

6장의 내용 : 파라미터들을 어떻게 갱신할 것인가? - 파라미터를 찾는 알고리즘(SGD, Momentum, AdaGrad, Adam)
            / 최적의 하이퍼 파라미터를 어떻게 찾을 것인가?
"""
from ch06.ex01_matplot3d import fn_derivative, fn
import numpy as np
import matplotlib.pyplot as plt


class Sgd:
    """
    SGD(Stochastic Gradient Descent, 확률적 경사 하강법)
    W = W - lr * dl/dw
    - W : 파라미터(가중치, 편향)
    - lr : 학습률(learning rate)
    - dl/dw : 변화율(gradient)   -> 오차역전파를 통해서 찾을 수 있음 (미분연쇄법칙 사용)
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate    # 파라미터에 저장된 learning_rate값을 field에 저장

    def update(self, params, gradients):
        """
        파라미터 params와 변화율 gradients를 이용해서
        파라미터들(W, b)을 갱신하는 메소드  - 갱신방법 : SGD
        - params, gradients : dict type
        """
        for key in params:
            params[key] -= self.learning_rate*gradients[key]


if __name__ == '__main__':
    sgd = Sgd(0.95)   # Sgd 클래스의 객체(인스턴스) 생성

    # ex01 모듈에서 작성한 fn(x, y) 함수의 최소값을 임의의 점에서 시작해서 찾기

    # 신경망에서 찾고자 하는 파라미터의 초기값
    init_position = (-7, 2)
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]

    # 각 파라미터에 대한 gradient
    gradient = dict()
    gradient['x'], gradient['y'] = 0, 0

    # 각 파라미터 갱신 값들을 저장할 리스트
    x_hist = []
    y_hist = []
    epoch = 30
    for i in range(epoch):
        x_hist.append(params['x'])
        y_hist.append(params['y'])
        gradient['x'], gradient['y'] = fn_derivative(params['x'], params['y'])
        sgd.update(params, gradient)

    for x, y in zip(x_hist, y_hist):
        print(f'({x, y})')

    # f(x, y) 함수를 등고선으로 표현
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    # mask = Z > 7
    # Z[mask] = 0        #-> z축이 7이상인 것은 0으로 표현해서 그래프에 그리지않겠다는 의미

    plt.contour(X, Y, Z, 30, cmap='binary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_hist, y_hist, 'o-', color='red')
    plt.show()



