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

#### 3) cross_correlation 함수
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

***********************************************************
## ex02_convolution_2d
### ※ 2차원 Convolution(합성곱), Cross_correlation(교차 상관) 연산
###### ex, x = [[1,2,3], [4,5,6], [7,8,9]], w = [[2,0], [0,0]] 인 경우
###### -> result = [[2, 4], [8, 10]]

#### 1) convolution 함수
```python
import numpy as np


def convolution_2d(x, w):
    """
    x, w: 2d ndarray, x.shape >= w.shape
    x와 w의 cross_correlation 결과 리턴
    convolution 결과 행렬의 shape :
    """
    conv = []
    xh, xw = x.shape[0], x.shape[1]
    wh, ww = w.shape[0], w.shape[1]
    result_row = xh - wh + 1
    result_col = xw - ww + 1

    for i in range(result_row):
        for j in range(result_col):
            x_sub = x[i:wh+i, j:ww+j]
            fma = np.sum(x_sub * w)
            conv.append(fma)
    conv = np.array(conv)
    return conv.reshape((result_row, result_col))
```

```python
if __name__ == '__main__':
    np.random.seed(113)
    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    print(x)
    print(w)
    print('convolution_2d\n', convolution_2d(x, w))   
```
###### x = [[5 2 2 6 7], [0 9 0 2 7], [4 5 5 6 1], [4 9 5 5 3], [7 7 5 0 1]]
###### w = [[1 1 3], [0 1 2], [3 1 0]]
###### convolution_2d =  [[39 46 66], [45 64 51], [71 69 40]]

***********************************************************
## ex03_padding
### ※ 패딩(padding)과 스트라이드(stride)
#### 1) 패딩 : 합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정값(ex, 0)으로 채우는 것
##### - Convolution을 하면 항상 x의 크기보다 작아지고 x의 원소마다 conv에 참여하는 횟수가 다름
##### - 따라서 입력데이터(x)의 주변을 특정 값으로 채워서 x의 모든 원소가 conv연산에 참여하는 횟수를 동일하게 만들어줌
##### ▷ 패딩의 목적 1) x의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding
##### ▷ 패딩의 목적 2) convolution 결과의 크기가 입력데이터 x와 동일한 크기가 되도록 padding

##### - 패딩을 어떻게 채우는가에 따라 결과가 달라질 수 있음

###### 2차원 ndarray에서 padding(row axis padding = 아래위, column axis padding = 양옆)
###### →row axis ->  before : 위, after : 아래
###### →column axis ->  before : 왼쪽, after : 오른쪽

##### -> numpy.pad()

###### ● pad_width : (before, after) : 축(axis)을 기준으로 앞(before)과 뒤(after)의 패딩 크기
###### → pad_width=1 : 앞, 뒤로 1개씩 padding 형성 -> 0123450
###### → pad_width=(2,3) : 앞에 2개, 뒤에 3개 padding 형성 -> 0012345000

###### ● mode : padding에 넣을 숫자 타입
###### → constant :  padding의 값을 상수로 함
###### → np.pad(x, pad_width=2, mode=’minimum’) -> 111234511

###### ● constant_value : 상수(constant)로 지정할 값


```python
if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 6)
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=(2, 3), mode='constant', constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=2, mode='minimum')
    print(x_pad)

    x = np.arange(1, 10).reshape((3, 3))
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=(1, 2), mode='constant', constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=((1, 2), (3, 4)), mode='constant', constant_values=0)
    print(x_pad)
```

***********************************************************
## ex04_conv_pad
### ※ 

***********************************************************
## ex05_image
### ※ 
