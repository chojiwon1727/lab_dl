"""
파라미터 최적화 알고리즘
4) Adam(Adaptive Moment estimate)
    - AdaGrade + Momentum알고리즘을 합친 것
    - 학습할 때 마다 학습률 변화  + 속도(모멘텀) 개념 도입
    W : 파라미터
    lr : 학습률(learning rate)
    t : 반복할 때마다 증가하는 숫자(timestamp)  ->  update 메소드가 호출될 때 마다 +1
    beta1, beta2 : 모멘텀을 변화시킬 때 사용하는 상수들
    m : 1st momentum
    v : 2nd momentum

    --> m = beta1 * m + (1-beta1) * gradient                    -> momentum 적용
        v = bata2 * v + (1-beta2) * gradient * gradient         -> momentum 적용
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W = W - [lr / (sqrt(v_hat))] * m_hat                         -> adagrad  적용
"""
import numpy as np
from ch06.ex01_matplot3d import fn, fn_derivative
import matplotlib.pyplot as plt

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.99):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = dict()
        self.v = dict()
        self.t = 0

    def update(self, params, gradients):
        self.t += 1

        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1-self.beta2) * gradients[key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= (self.lr / np.sqrt(v_hat)) * m_hat


if __name__ == '__main__':
    params = {'x': -7.0, 'y': 2.0}  # 파라미터 초깃값
    gradients = {'x': 0.0, 'y': 0.0}  # gradient 초깃값

    # Adam 클래스의 인스턴스 생성
    adam = Adam(lr=0.3)  # lr=0.01, 0.1, 0.3

    # 학습하면서 파라미터(x, y)들이 업데이트되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)
        # 변경된 파라미터 값 출력
        print(f"({params['x']}, {params['y']})")

    # 등고선(contour) 그래프
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam')
    plt.axis('equal')
    # x_history, y_history를 plot
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()







