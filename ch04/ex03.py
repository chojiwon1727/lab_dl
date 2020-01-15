"""
교차엔트로피:
entropy = -true_value * log(expected_value)
entropy = -sum i [t_i * log(y_i)]
"""
import pickle
import numpy as np
from ch03.ex11 import forward
from dataset.mnist import load_mnist


def _cross_entropy(y_pred, y_true):
    delta = 1e-7            # log0(-무한대)가 되는 것을 방지하기 위해서 더해줄 값(scala - Broadcast 됨)
    return -np.sum(y_true * np.log(y_pred+delta))


def cross_entropy(y_pred, y_true):
    if y_pred.ndim == 1:   # 데이터 개수 1개
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true) / len(y_pred)
    return ce


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)
    y_true = y_test[:10]
    X_true = X_test[:10]

    with open('../ch03/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    y_pred = forward(network, X_true)

    # 원본과 예측이 같은 경우
    print('y_true[0]:', y_true[0])   # 숫자 7 이미지
    print('y_pred[0]:', y_pred[0])   # 7 이미지가 될 확률이 가장 큼
    print('ce:', cross_entropy(y_pred[0], y_true[0]))     # 0.00293918838724494

    # 원본과 예측이 다른 경우
    print('y_true[8]:', y_true[8])   # 숫자 5 이미지
    print('y_pred[8]:', y_pred[8])   # 6 이미지가 될 확률이 가장 큼
    print('ce:', cross_entropy(y_pred[8], y_true[8]))     # 4.909424304962158
    print('2차원 ce:', cross_entropy(y_pred, y_true))     # 0.5206955424044282

    # 만약 y_true 또는 y_pred가 one-hot encoding이 사용되지 않으면,
    # one-hot encoding 형태로 변환해서 cross-entropy 계산
    np.random.seed(1227)
    y_true = np.random.randint(10, size=10)
    print('y_true:', y_true)
    y_true2 = np.zeros((y_true.size, 10))   # 원소 개수만큼 row, 숫자는 0~9까지니까 10개
    for i in range(y_true.size):
        y_true2[i][y_true[i]] = 1
    print(y_true2)

