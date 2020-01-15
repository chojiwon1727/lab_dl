"""
Machine Learning(기계 학습) -> Deep Learning(심층 학습)
training data set(학습 세트) / test data set(검증 세트)
- 손실함수
1) 평균제곱오차
2) 교차엔트로피
"""
import math

from dataset.mnist import load_mnist
import numpy as np


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()
    print('y_true[:10] =', y_true[:10])   # 10개 테스트 이미지들의 실제 값

    # 10개 테스트 데이터 이미지들의 예측값
    y_pred = np.array([7, 2, 1, 6, 4, 1, 4, 9, 6, 9])
    print('y_pred[:10] =', y_pred[:10])   # 10개 테스트 이미지들의 예측 값

    # 오차
    error = y_pred - y_true[:10]
    print('error:', error)

    # 오차 제곱(Squared Error)
    sq_err = error**2
    print('Squared Error:', sq_err)

    # 평균 오차 제곱(Mean Squared Error)
    MSE = np.mean(sq_err)
    print('MSE:', MSE)

    # RMSE
    RMSE = math.sqrt(MSE)
    print('RMSE:', RMSE)
