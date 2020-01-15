# 책 82p 그림 3-14(포스트잇 그림)
import numpy as np

x = np.array([1,2])
W = np.array([[1,4],
              [2,5],
              [3,6]])
b = 1
y = W.dot(x) + b   # bias를 행렬로 만들어도 되지만 그냥 더해도 똑같은 효과가 있음
print(y)

x = np.array([1,2])
W = np.array([[1,2,3],
             [4,5,6]])
b = 1
y = x.dot(W) + b
print(y)