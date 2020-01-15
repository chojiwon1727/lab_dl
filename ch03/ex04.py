import numpy as np
from ch03.ex01 import sigmoid

x = np.array([1, 2])
W_1 = np.array([[1,2,3],
              [4,5,6]])
b_1 = np.array([1,2,3])

a_1 = x.dot(W_1) + b_1
print('a_1 :', a_1)

z_1 = sigmoid(a_1)
print('z_1 :', z_1)

# 두번째 은닉층에 대한 가중치 행렬 w_2와 bias 행렬 b_2 작성
w_2 = np.array([[0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6]])
b_2 = np.array([0.1, 0.2])

a_2 = z_1.dot(w_2) + b_2
print('a_2 :', a_2)

y = sigmoid(a_2)
print('y :', y)