"""
경사하강법(gradient descent)
x_new = x lr * df/dx
위 과정을 반복 -> f(x)의 최소값을 찾음
"""
import numpy as np
import matplotlib.pyplot as plt
from ch04.ex05 import numerical_gradient


def gradient_method(fn, x_init, lr=0.01, step=100):
    """
    경사 하강법 : 현 위치에서 기울어진 방향으로 일정 거리만큼 이동,
                 그런 다음 이동한 곳에서도 마찬가지로 기울기를 구하고,
                 또 그 기울어진 방향으로 나아가기를 반복해서 함수의 값을 점차 줄이는 것
    :param fn: 그래프를 그릴 함수
    :param x_init: 초기 x값
    :param lr: 학습율(변화율)
    :param step: 반복횟수
    :return: 최종 x값
    """
    x = x_init          # 점진적으로 변화시킬 변수(초기 x값)
    x_history = []      # x가 변화되는 과정을 저장할 배열
    for i in range(step):
        x_history.append(x.copy())              # x의 복사본을 x 변화 과정에 기록
        gradient = numerical_gradient(fn, x)    # 점 x에서의 gradient 계산
        x -= lr * gradient   # x_new = x_init - lr * gradient -> x를 변경
    return x, np.array(x_history)


def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    init_x = np.array([4.0])
    x, x_hist = gradient_method(fn, init_x, lr=0.1)
    print('x =', x)
    print('x_history =', x_hist)

    init_x = np.array([-4., 3.])
    x, x_hist = gradient_method(fn, init_x, lr=0.1, step=100)
    print('x =', x)
    print('x_history =', x_hist)

    # x_hist(x좌표가 변경되는 과정)을 산점도로 그리기
    plt.scatter(x_hist[:, 0], x_hist[:, 1])

    # 동심원 : x**2 + y**2 = r**2     -> y**2 = r**2 - x**2
    for r in range(1,5):
        r = float(r)    # 정수 -> 실수 변환
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)
        y_pts2 = -np.sqrt(r**2 - x_pts**2)
        plt.plot(x_pts, y_pts1, ':', color='gray')
        plt.plot(x_pts, y_pts2, ':', color='gray')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(color='0.8')
    plt.axvline(color='0.8')
    plt.show()