"""
MNIST 숫자 손글씨 데이터 세트
"""
from PIL import Image
import numpy as np
from dataset.mnist import load_mnist


def img_show(img_arr):
    """
    numpy 배열(ndarray)로 작성된 이미지를 화면 출력
    Numpy 배열 형식을 이미지로 변환
    """
    img = Image.fromarray(np.uint8(img_arr))    # -> np.uint8 : 부호가 없는 8bit(2의 8승, 0~2^8-1 = 0~255까지 저장)
    img.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=False,)
    # (학습 이미지 데이터, 학습 데이터 레이블), (테스트 이미지 데이터, 테스트 데이터 레이블)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    # 학습 세트의 첫번째 이미지
    img = X_train[0]
    img = img.reshape((28,28))    # 1차원 배열을 28x28 형태의 2차원 배열로 변환
    print(img)
    img_show(img)   # 2차원 numpy 배열을 이미지로 출력
    print('label:', y_train[0])

    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('y_train[0] :', y_train[0])
    img = X_train[0]
    print(img)