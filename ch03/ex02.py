"""
행렬의 내적(dot product)

대문자 변수 : 2차원 이상의 ndarray
소문자 변수 : 1차원 ndarray
"""
import numpy as np

x = np.array([1,2])
W = np.array([[3,4],
             [5,6]])
print(x.dot(W))

A = np.arange(1, 7).reshape((2,3))
B = np.arange(1,7).reshape((3,2))
print(A.dot(B))
print(B.dot(A))
print()

# ndarray.shape -> 1차원 : (x, ) - 원소의 개수 / 2차원 : (x, y) / 3차원(x, y, z), ...
x = np.array([1,2,3])
print(x)
print(x.shape)    # -> (3,)

x = x.reshape((3,1))    # -> column vector
print(x)
print(x.shape)    # -> (3, 1)

x = x.reshape((1,3))   # -> row vector
print(x)
print(x.shape)    # -> (1, 3)
