from ch03.ex05 import softmax
import numpy as np
if __name__ == '__main__':

    # softmax에 1차원 배열을 넘긴 경우 -> [0.09003057 0.24472847 0.66524096]
    x = np.array([1,2,3])
    s = softmax(x)
    print(s)

    # softmax에 2차원 배열을 넘긴 경우 : 배열을 따로
    # softmax함수를 만들 때 배열의 최대값을 구하는데,np.max(2차원 배열)을 하면
    # max값을 구하는 방식이 3가지임! -> 전체 원소의 최대값, 각 row별 최대값, 각 column별 최대값
    #                                -> np.max(arr, axis=)
    X = np.array([[1,2,3],
                  [4,5,6]])
    s = softmax(X)
    print(s)

    # Numpy Broadcast
    x = np.array([1,2,3])
    print('x =', x)
    print('x + 10=', x + 10)

    X = np.arange(6).reshape((2,3))
    print('X shape:', X.shape)
    print('X \n', X)
    a = np.arange(1,4)
    print('a shape:', a.shape)
    print('a \n', a)
    print('X + a \n',X + a)

    b = np.array([10,20])
    b = b.reshape((2,1))
    print('b shape:', b.shape)
    print('b \n', b)
    print('X + b \n',X + b)

    print('====================================================================')
    np.random.seed(2020)
    X = np.random.randint(10, size=(2, 3))
    print(X)
    # 1. X의 모든 원소들 중 최댓값(m)을 찾아서,
    # X - m을 계산해서 출력
    m = np.max(X)
    print(f'm = {m}')
    result = X - m  # (2,3) shape의 2차원 배열과 스칼라 간의 broadcast
    print(result)

    # 2. X의 axis=0 방향의 최댓값들(각 컬럼에서의 최댓값들)을 찾아서,
    # X의 각 원소에서, 그 원소가 속한 컬럼의 최댓값을 뺀 행렬을 출력
    m = np.max(X, axis=0)
    print(f'm = {m}, shape: {m.shape}')
    result = X - m
    print(result)

    # 3. X의 axis=1 방향의 최댓값들(각 row에서의 최댓값들)을 찾아서,
    # X의 각 원소에서, 그 원소가 속한 row의 최댓값을 뺀 행렬을 출력
    m = np.max(X, axis=1)
    print(f'm = {m}, shape: {m.shape}')
    result = X - m.reshape((2, 1))
    print(result)

    X_t = X.T  # transpose: 행렬의 행과 열을 바꾼 행렬
    print(X_t)
    m = np.max(X_t, axis=0)  # 전치 행렬에서 axis=0 방향으로 최댓값 찾음.
    result = X_t - m  # 전치 행렬에서 최댓값들을 뺌.
    result = result.T  # 전치 행렬을 다시 한번 transpose
    print(result)

    # 4. X의 각 원소에서, 그 원소가 속한 컬럼의 최댓값을 뺀 행렬의
    # 컬럼별 원소들의 합계
    m = np.max(X, axis=0)
    result = X - m
    s = np.sum(result, axis=0)
    print(s)

    # 5. X의 각 원소에서, 그 원소가 속한 row의 최댓값을 뺀 행렬의
    # row별 원소들의 합계
    m = np.max(X, axis=1)
    result = X - m.reshape((2, 1))
    s = np.sum(result, axis=1)
    print(s)

    # 표준화: 평균 mu, 표준편차 sigma -> (x - mu)/sigma
    # axis=0 방향으로 표준화
    print(X)
    X_new = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(X_new)

    # axis=1 방향으로 표준화
    mu = np.mean(X, axis=1).reshape((2, 1))
    sigma = np.std(X, axis=1).reshape((2, 1))
    X_new = (X - mu) / sigma
    print(X_new)

    # transpose 이용
    X_t = X.T
    X_new = (X_t - np.mean(X_t, axis=0)) / np.std(X_t, axis=0)
    X_new = X_new.T
    print(X_new)