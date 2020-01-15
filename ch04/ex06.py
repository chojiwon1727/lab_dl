import numpy as np
from ch04.ex05 import numerical_gradient
import matplotlib.pyplot as plt

def fn(x):
    """ x = [x0, x1] """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


x0 = np.arange(-1, 2)
print('x0 =', x0)

x1 = np.arange(-1, 2)
print('x1 =', x1)

X, Y = np.meshgrid(x0, x1)
print('X =', X)
print('Y =', Y)

# 2차원 배열인 X와 Y를 합치기 전에 각 2차원 배열을 1차원 배열로 변경(변경 후 더하면 XY의 2차원 배열이 됨)
X = X.flatten()
Y = Y.flatten()
print('X =', X)
print('Y =', Y)

XY = np.array([X, Y])
print('XY =', XY)

gradient = numerical_gradient(fn, XY)
print('gradient = ', gradient)

# 책 126 그래프 그리기
print()
x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)

X, Y = np.meshgrid(x0, x1)
X = X.flatten()  # 1차원 배열
Y = Y.flatten()  # 1차원 배열
XY = np.array([X, Y])  # 2차원 배열
gradient = numerical_gradient(fn, XY)
# print('gradient = ', gradient)

plt.quiver(X, Y, -gradient[0], -gradient[1], angles='xy')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()

