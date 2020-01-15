"""
파라미터 최적화 알고리즘
3) AdaGrad(Adaptive Gradient)
    - SGD : W = W - lr * gradient
    - epoch 실행 중에 lr(학습률)을 변경시켜가는 것
    - 처음에는 큰 학습률로 시작해서 점점 학습률을 줄여나가면서 최적의 파라미터를 찾는 것
    h = h + gradient * gradient
    lr = lr / sqrt(h)    -> gradient가 커진다는 가정 아래 반복할때마다 lr이 점점 작아질 것
    W = W - (lr / sqrt(h)) * gradient
"""
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01_matplot3d import fn, fn_derivative


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = dict()

    def update(self, params, gradients):
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])

        for key in params:
            self.h[key] += gradients[key] * gradients[key]
            epsilon = 1e-8   # 0으로 나누는 것을 방지하기 위함
            params[key] -= (self.lr / np.sqrt(self.h[key] + epsilon)) * gradients[key]


if __name__ == '__main__':
    adagrad = AdaGrad(lr=1.5)
    params = {'x': -7.0, 'y': 2.0}
    gradients = {'x': 0.0, 'y': 0.0}

    x_hist = []
    y_hist = []
    epoch = 30
    for i in range(epoch):
        x_hist.append(params['x'])
        y_hist.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adagrad.update(params, gradients)

    for x, y in zip(x_hist, y_hist):
        print(f'({x}, {y})')

    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('AdaGrad')

    plt.plot(x_hist, y_hist, 'o-', color='red')
    plt.show()