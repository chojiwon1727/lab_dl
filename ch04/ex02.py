import pickle

import numpy as np

from ch03.ex11 import forward
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (X_train, y_train),(X_test, y_test) = load_mnist(one_hot_label=True)

    X_true = X_test[:10]
    y_true = y_test[:10]
    print('y_true:', y_true[0])
    print('X_true:', X_true[0])

    with open('../ch03/sample_weight.pkl', 'rb')as f:
        network = pickle.load(f)
    y_pred = forward(network, X_true)   # (10, 10)
    print('y_pred:', y_pred)

    print(y_true[0])
    print(y_pred[0])

    error = y_pred[0] -y_true[0]
    print(error)
    print(error**2)
    print(np.sum(error**2))

    print('y_true[8] :', y_true[8])   # [:10]까지 중 틀린 인덱스 : 8
    print('y_pred[8] :', y_pred[8])
    print(np.sum((y_true[8] - y_pred[8])**2))    # 1.888963693926786  -> 이 값을 줄이는 것이 좋음
