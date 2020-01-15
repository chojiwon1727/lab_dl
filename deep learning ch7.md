# ch07. 합성곱 신경망(Convolutional neural network, CNN)
#### ※합성곱 신경망
##### - 6장까지의 신경망은 완전연결(fully connected)라고 하며, 완전히 연결된 계층을 Affine계층이라는 이름으로 구현
##### - 완전 연결 계층은 Affine 계층 뒤에 활성화 함수 ReLU(혹은 Sigmoid)가 이어지지만, 
##### - CNN으로 이루어진 네트워크에서는 새로운 ‘합성곱 계층(Conv)’과 ‘풀링 계층(Pooling)’이 추가됨
###### -> **완전 연결 계층**의 은닉층 : 'Affine -> ReLU'
######      ●  완전 연결 계층 : X -> [Affine -> ReLU] -> ... -> [Affine -> Softmax] 
######      완전 연결 계층의 문제점 : 데이터의 형상이 무시됨(ex, 3차원 데이터를 평평한 1차원으로 변형해서 사용)
###### -> **CNN**의 은닉층 : 'Conv -> ReLU -> (Pooling)'
######      ●  CNN : X -> [Conv -> ReLU -> (Pooling)] -> ... -> [Affine -> Softmax] 

## ex01_convolution1d
#### ※ 1. 합성곱 계층 설명
##### --> CNN : X -> [Conv -> ReLU -> (Pooling)] -> ... -> [Affine -> Softmax]
##### 1) 합성곱연산(x conv w)
###### ①  w반전(반전有 : Convolution, 반전無 : Cross-Correlation)
###### ②  w를 한 칸씩(보폭, stride = 1)밀면서 원소별 곱하기 & 더하기 -> FMA(Fused Multiply-Add)

###### ex, x = [1,2,3,4,5], w = [2,1] 인 경우
###### ① w = [1,2]
###### ② 1x1 + 2x2 = 5, 2x1 + 3x2 = 8, 3x1 + 4x2 = 11, 4x1 + 5x2 = 14 -> result = [5, 8, 11, 14]

#### 2) convolution함수
```python
import numpy as np


def convolution_1d(x, w):
    """ x, w : 1d ndarray, len(x) >= len(w) """
    w_r = np.flip(w)
    conv = []
    len_result = len(x) - len(w) + 1
    for i in range(len_result):
        x_sub = x[i:i+len(w)]
        fma = np.sum(x_sub * w_r)
        conv.append(fma)
    conv = np.array(conv)
    return conv
```

#### 2) cross_correlation 함수
```python
def cross_correlation_1d(x, w, convolution=False):
    """ x, w : 1d ndarray, len(x) >= len(w) """
    if convolution == True:
        w = np.flip(w)

    conv = []
    len_result = len(x) - len(w) + 1
    for i in range(len_result):
        x_sub = x[i:i + len(w)]
        fma = np.sum(x_sub * w)
        conv.append(fma)
    conv = np.array(conv)
    return conv
```

```python
if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    w = np.array([2, 1])
    w_r = np.flip(w)

    conv = []
    for i in range(4):
        x_sub = x[i:i+len(w)]   # (0,1), (1,2), (2,3), (3,4)
        fma = np.sum(x_sub * w_r) # 1차원인 경우, np.dot(x_sub, w_r) 동일
        conv.append(fma)
    conv = np.array(conv)
    print('conv =', conv)
    print('conv =', convolution_1d(x, w))

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    print('conv =', convolution_1d(x, w))

    cross_correlation = cross_correlation_1d(x, w)
    print('cross_correlation =', cross_correlation)
```
##### --> 교차상관(Cross-correlation)과 합성곱 연산이 다른 점은 w를 반전시키지 않는 것
##### --> CNN(Convolutional Neural Network, 합성곱 신경망)에서는 대부분 교차상관 사용
