"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle
from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist
import numpy as np


def init_network():
    """ 가중치, bias행렬들을 생성 """
    with open('sample_weight.pkl', 'rb') as f: # 교재의 저자가 만든 가중치 행렬을 읽어옴
        network = pickle.load(f)
    # W1, W2, W3, b1, b2, b3 shape 확인
    return network


def predict(network, X_test):
    """
    신경망에서 사용되는 가중치행렬과 테스트 데이터를 전달받아서
    테스트 데이터의 예측값(배열)을 리턴
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = X_test.dot(W1) + b1
    z1 = sigmoid(a1)

    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    pred = z2.dot(W3) + b3
    return softmax(pred)


def accuracy(y_test, pred):
    """
    테스트 데이터레이블과 테스트 데이터 예측값을 전달받아서
    정확도(accuracy)= (정답/전체) 를 리턴
    """
    acc = []
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] == 1:
                a, b = i, j
                if pred[i][j] == np.max(pred[i]):
                    c, d = i, j
                    if a == c and b == d:
                        acc.append(a)
        return len(acc) / len(y_test)


if __name__ == '__main__':
    network = init_network()

    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    pred = predict(network, X_test)
    print('pred[0]:',pred[0])
    print('max:',np.max(pred[0]))
    print(y_test[0])


    acc = accuracy(y_test, pred)
    print(acc)