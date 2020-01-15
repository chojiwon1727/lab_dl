import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d     # 3D 그래프를 그리기 위해 반드시 필요함
import numpy as np


def fn(x, y):
    """ f(x, y) = (1/20) * x**2 + y**2 """
    return x**2 / 20 + y**2


def fn_derivative(x, y):
    """ fn(x, y)함수의 미분값 리턴 -> 함수의 변화율 """
    return x/ 10, 2*y


if __name__ == '__main__':
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)

    # 3차원 그래프를 그리기 위해서는 x좌표와 y좌표의 쌍으로 이루어진 데이터가 필요함
    # -> meshgrid 두 배열을 합쳐서 행렬을 만들어줌
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')    # projection 파라미터 사용을 위해서 import mpl_toolkits.mplot3d 필요
    plt.xlabel('x')
    plt.ylabel('y')

    ax.contour3D(X, Y, Z, 100)     # -> 숫자: X, Y를 Z축으로 얼마나 촘촘하게 쌓을것인가 / cmap 파라미터를 사용하면 색 변경가능
    plt.show()


    # 3d 그래프를 z축의 위쪽에서 바라본 것 : 등고선(contour) 그래프
    # 등고선 그래프도 meshgrid를 사용한 X, Y, Z가 필요함
    plt.contour(X, Y, Z, 50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    # 그래프가 촘촘할수록 경사가 급하다는 의미
    # -> 경사가 급한쪽의 임의의 점에서 최소점을 찾아가면 금방 찾아가지만, 경사가 완만한 쪽의 임의의 점에서는 시간이 오래 걸림

