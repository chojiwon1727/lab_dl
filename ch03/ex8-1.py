"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

from PIL import Image

from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist
import numpy as np


def init_network():
    """ 가중치, bias행렬들을 생성 """
    with open('sample_weight.pkl', 'rb') as f:  # 교재의 저자가 만든 가중치 행렬을 읽어옴
        network = pickle.load(f)
    # W1, W2, W3, b1, b2, b3 shape 확인
    return network


def forward(network, x):
    """
    x : 이미지 한 개의 정보를 가지고 있는 배열(784, )
    """
    # 가중치, 편향 행렬
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)

    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    a3 = z2.dot(W3) + b3
    y = softmax(a3)
    return y


def predict(network, X_test):
    """
    신경망에서 사용되는 가중치행렬과 테스트 데이터를 전달받아서
    테스트 데이터의 예측값(배열)을 리턴
    X_test : 10,000개의 테스트 이미지들의 정보를 가지고 있는 배열
    """
    pred = []
    for sample in X_test:
        # 이미지를 신경망에 전파(통과) 시켜서 어떤 숫자가 될 지 확률 계산
        sample_hat = forward(network, sample)
        # 가장 큰 확률의 인덱스를 찾음
        y_pred = np.argmax(sample_hat)
        pred.append(y_pred)  # 예측값을 결과 list에 추가
    return pred


def accuracy(y_test, pred):
    """
    테스트 데이터레이블과 테스트 데이터 예측값을 전달받아서
    정확도(accuracy)= (정답/전체) 를 리턴
    """
    result = (y_test == pred)
    return np.mean(result)   # np.mean(array(bool)) -> True를 1로, False를 0으로 치환해서 계산


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    # 신경망 가중치와 편향 행렬 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print('W1 =', W1.shape)
    print('W2 =', W2.shape)
    print('W3 =', W3.shape)
    print('b1 =', b1.shape)
    print('b2 =', b2.shape)
    print('b3 =', b3.shape)

    pred = predict(network, X_test)
    print(pred[:10])
    print(y_test[:10])

    acc = accuracy(y_test, pred)
    print(acc)

    # 예측이 틀린 첫번째 이미지: X_test[8]    ->  normalize되어 있고, 1차원 배열로 flatten되어 있어서 볼 수 없음
    img = X_test[8] * 255    # 0~1 -> 0~255(denormalize)
    img = img.reshape((28,28))   # 1차원 배열  -> 2차원 배열
    img = Image.fromarray(img)
    img.show()
