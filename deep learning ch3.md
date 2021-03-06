# ch03. 신경망
## ex01
### 활성화 함수(activation function)
### : 신경망의 뉴런(neuron)에서 입력 신호의 가중치 합을 출력값으로 변환해주는 함수
#### ※ perceptron 
#### - 입력 : (x1, x2)
#### - 출력 : a = x1*w1 + x2*w2 + b
####        -> y = 1, a > 임계값
####        -> y = 0, a <= 임계값  -> a값이 크다, 작다를 결정하는 것이 활성화 함수
####        -> 분류문제면 1, 0이 나오고, 회귀문제이면 연속된 결과를 줄 것!

#### 활성화함수 1) 계단함수
```python
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
```

```python
def step_function2(x):
    """
    Step Function

    :param x: numpy.ndarray
    :return: step 함수 출력(0,1)로 이루어진 numpy.ndarray
    """
    # result = []
    # for x_i in x:
    #     if x_i > 0:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return np.array(result)

    y = x > 0   # np.ndarray 자체가 원소간 비교를 하기 때문에 이 자체에 for문이 들어간 것과 같음(F, T의 array) -> F, T를 0,1로만 바꾸면 됨
    return y.astype(np.int)   # ndarray.astype(np.int)   -> array의 다른 타입을 int타입으로 변환해주는 함수

```

```python
if __name__ == '__main__':
x = np.arange(-3, 4)     # 파이썬의 range()함수와 유사(시작값부터 끝나는 값까지 1씩 증가시키는 함수)
    print('x = ', x)

    for x_i in x:                     # array의 값을 넘기는 것
        print(step_function(x_i), end='')
    print()

    print('y = ', step_function2(x))     # array의 값을 넘기는 것이 아니라 array 자체를 넘기면 array를 리턴해주면 좋겠음
```
###### x =  [-3 -2 -1  0  1  2  3]
###### 0000111
###### y =  [0 0 0 0 1 1 1]

#### 활성화함수 2) sigmoid함수
```python
def sigmoid(x):
    """
    sigmoid = 1 / (1 + exp(-x))

    * exp *
    math.exp : x가 number만 가능
    np.exp : x가 number, ndarray, iterable 다 가능
    """
    return 1 / (1 + np.exp(-x))
```

```python
if __name__ == '__main__':
print('sigmoid : ', sigmoid(x))

    x_points = np.linspace(-10, 10, 100)
    step = step_function2(x_points)
    sigmoid = sigmoid(x_points)

    plt.plot(x_points, step, label='step function')
    plt.plot(x_points, sigmoid, label='sigmoid function')
    plt.legend()
    plt.show()
```
###### 
![Figure_1](https://user-images.githubusercontent.com/56914237/72419218-b4cf7f00-37bf-11ea-88a1-ff81d85c7b73.png)

#### 활성화함수 3) ReLU함수
```python
def relu(x):
    """
    RuLU(Rectified Linear Unit)
     y = x, if x > 0
       = 0, otherwise(x < 0이면 0으로 표현)
    """
    result = []
    for x_i in x:
        if x_i > 0:
            result.append(x_i)
        else:
            result.append(0)
    return np.array(result)

    # result = [x_i if x_i > 0 else 0 for x_i in x]
    # return np.array(result)
    # return np.maximum(0, x)
```

```python
if __name__ == '__main__':
relu = relu(x_points)
    print(relu)
    plt.plot(x_points, relu, label='relu function')
    plt.legend()
    plt.show()
```
![Figure_2](https://user-images.githubusercontent.com/56914237/72419963-31169200-37c1-11ea-88b0-ef1884571110.png)

***********************************************************
## ex02
### 행렬의 내적(dot product)
```python
import numpy as np

x = np.array([1,2])
W = np.array([[3,4],
             [5,6]])
print(x.dot(W))
```
###### [13 16]

```python
A = np.arange(1, 7).reshape((2,3))
B = np.arange(1,7).reshape((3,2))
print(A.dot(B))
print(B.dot(A))
```
###### [[22 28], [49 64]] 
###### [[ 9 12 15], [19 26 33], [29 40 51]]
###### -> W.dot(x)하면 원래 dot product를 할 수 없는데 numpy가 자동으로 x(1x2)를 2x1로 변형 계산

***********************************************************
## ex03
### 행렬을 이용한 신경망

#### 신경망1 -> 포스트잇 그림 1 추가하기(책82p 그림 3-14)
```python
import numpy as np

# x1이 1, x2가 2인 경우
x = np.array([1,2])
W = np.array([[1,4],
              [2,5],
              [3,6]])
b = 1
y = W.dot(x) + b   # bias를 행렬로 만들어도 되지만 그냥 더해도 똑같은 효과가 있음
print(y)
```
###### [10 13 16]

```python 
x = np.array([1,2])
W = np.array([[1,2,3],
             [4,5,6]])
b = 1
y = x.dot(W) + b
print(y)
```
###### [10 13 16]

***********************************************************
## ex04
#### 신경망2 -> 포스트잇 그림 2 추가하기 
##### 그림 2-1
##### 1) 첫번째 은닉층 a_1 계산해서 은닉층 a_1의 a값들의 리스트 출력
```python
import numpy as np
from ch03.ex01 import sigmoid

x = np.array([1, 2])
W_1 = np.array([[1,2,3],
              [4,5,6]])
b_1 = np.array([1,2,3])

a_1 = x.dot(W_1) + b_1
print('a_1 :', a_1)
```
###### a_1 : [10 14 18]

##### 2) 은닉층 a_1을 활성화 함수 sigmoid 함수로 적용한 값 z_1의 리스트 출력 
```python
z_1 = sigmoid(a_1)
print('z_1 :', z_1)
```
###### z_1 : [0.9999546  0.99999917 0.99999998]

##### 그림 2-2
##### 1) 두번째 은닉층 a_2를 계산해서 은닉층 a_2의 a값들의 리스트 출력
```python
w_2 = np.array([[0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6]])
b_2 = np.array([0.1, 0.2])

a_2 = z_1.dot(w_2) + b_2
print('a_2 :', a_2)
```
###### a_2 : [0.69999529 1.69998142]

##### 2) 은닉층 a_2를 활성화 함수 sigmoid 함수로 적용한 값 y의 리스트 출력
```python
y = sigmoid(a_2)
print('y :', y)
```
###### y : [0.66818673 0.84553231]
###### --> 그림에서 y1 : 0.668, y2 : 0.845

#### ==> 딥러닝의 목표 : y1, y2의 오차를 가장 작게 만드는 가중치(W)를 찾는 것
#### (방법 : 오차역전파, 경사하강법 등)

***********************************************************
## ex05
### 신경망 모델 코드로 구현하기
![캡처2](https://user-images.githubusercontent.com/56914237/72421871-a899f080-37c4-11ea-9e7e-f3fddf4a1f79.PNG)

#### 신경망에서 사용되는 가중치(W)행렬과 편향(b)행렬 생성
```python 
import numpy as np
from ch03.ex01 import sigmoid


def init_network():
    """
    신경망(neural network)에서 사용되는 가중치(W) 행렬과 bias 행렬 생성
    입력값 : (x1, x2) -> 1x2 행렬
    은닉층 : 2개
            --> 1st 은닉층 : 뉴런 3개  (x @ w1 + b1)
            --> 2nd 은닉층 : 뉴런 2개 (z @ w2 + b2)
    출력층 : (y1, y2) -> 1x2 행렬
    w1, w2, w3, b1, b2, b3를 난수로 생성
    """
    np.random.seed(1224)
    network = dict()       # 가중치 / bias 행렬을 저장하기 위한 dict(key : w1, value: 난수) -> 최종 리턴값

    # 은닉층 1 : x @ w1 + b1
    # (1 x 2) @ (2 x 3) + (1 x 3)  =  (1 x 3)
    network['W1'] = np.random.random(size=(2,3)).round(2)     # -> size를 주면 0~1사이의 값을 size에 맞게 리턴
    network['b1'] = np.random.random(size=3).round(2)

    # 은닉층 2 : z1 @ w2 + b2
    # (1 x 3) @ (3 x 1) + (1 x 2)  =  (1 x 2)
    network['W2'] = np.random.random(size=(3, 2)).round(2)
    network['b2'] = np.random.random(size=2).round(2)

    # 출력층 : z2 @ W3 + b3
    # (1 x 2) @ (2 x 2) + (1 x 2)  =  (1 x 2)
    network['W3'] = np.random.random(size=(2, 2)).round(2)
    network['b3'] = np.random.random(size=2).round(2)

    return network
```

#### 신경망에서 입력층 -> 은닉층 -> 출력층을 거치는 순방향 전파
```python
def forward(network, x):
    """
    순방향 전파(forward propagation) : 입력층 -> 은닉층 -> 출력층

    은닉층의 활성화 함수 : 시그모이드 함수

    :param network: 신경망에서 사용되는 가중치와 bias 행렬들을 저장한 dict
    :param x: 입력값을 가지고 있는 1차원 리스트  -> ex, [x1, x2]
    :return: 2개의 은닉층과 1개의 출력층을 거친 후 계산된 최종 출력값 -> ex, [y1, y2]
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1)+b1
    z1 = sigmoid(a1)

    z2 = sigmoid(z1.dot(W2)+b2)

    y = z2.dot(W3)+b3
    return softmax(y)
```

#### 출력층의 활성화 함수 1) 항등함수 : 회귀문제
```python
def identity_function(x):
    return x
```

#### 출력층의 활성화 함수 2) softmax : 분류문제
##### -> softmax의 문제점 : 지수승을 사용하기 때문에 x의 값이 커지면 급격히 증가(overfitting)
##### -> 따라서, softmax 공식에서 분자 분모에서 exp안에 똑같은 값을 더하거나 빼면 제대로된 결과를 얻을 수 있음

```python
def softmax(x):
    """
    x = [x1, x2, x3, ..., x_k, ..., x_n]
    배열 x의 k번째 값의 softmax y_k = exp(x_k) / [sum k to n exp(x_k)] = exp(x_k) / sum(exp(x_k))
    --> 의미 : 배열 x에서 k가 차지하는 비율(확률) --> 0과 1사이의 값
    :param x: array
    :return: 0 ~ 1사이의 값(모든 리턴값의 총 합 : 1)
    """
    max_x = np.max(x)  # array x의 원소들 중 최대값을 찾음
    y = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return y
```

```python
if __name__ == '__main__':
    network = init_network()

    x = np.array([1,2])
    y = forward(network, x)
    print('y :', y)

    # print('x =', x)
    # print('softmax =', softmax(x))
    #
    # x = [1,2,3]
    # print('softmax =', softmax(x))
    #
    # x = [1e0, 1e1, 1e2, 1e3]   # [1, 10, 100, 1000]
    # print('x =', x)
    print('softmax =', softmax(x))
```
###### y : [0.39250082 0.60749918]
###### softmax = [0.26894142 0.73105858]

***********************************************************
## ex06
### ※ MNIST database
http://yann.lecun.com/exdb/mnist/

#### load_mnist()
###### ●  파라미터 1) normalize 
###### -> normalize = True인 경우, 실제 배열의 값을 0 ~ 1 사이의 값으로 만들어줌

##### ●  파라미터 2) flatten 
###### -> flatten = False인 경우, 이미지 구성(n, c, h, w)형식으로 표시함
###### -> ex, (60000, 1, 28, 28) : 60_000개의 이미지가 모두 한 장(흑백=1, 컬러=3, 컬러+투명도=4)의 28x28로 구성

##### ●  파라미터 3) one_hot_label 
###### -> one_hot_label = True인 경우, one_hot_encoding 형식으로 숫자 레이블 출력
###### -> one_hot_encoding : 해당 자릿수만 1, 나머지 0 표시
###### -> ex, 5 = [0 0 0 0 0 1 0 0 0 0]

##### 신경망에 넣을 때 normalize=True, flatten=True가 좋음(but, 이미지 한 장을 보기에는 flatten=False가 좋음)
```python
from PIL import Image
import numpy as np
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=False)
    # (학습 이미지 데이터, 학습 데이터 레이블), (테스트 이미지 데이터, 테스트 데이터 레이블)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    # 학습 세트의 첫번째 이미지
    img = X_train[0]
    img = img.reshape((28,28))    # 1차원 배열을 28x28 형태의 2차원 배열로 변환
    print(img)
    img_show(img)   # 2차원 numpy 배열을 이미지로 출력
    print('label:', y_train[0])
```

```python
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('y_train[0] :', y_train[0])
    img = X_train[0]
    print(img)
```

***********************************************************
## ex07
### ※ pickle 데이터 타입
#### - Serialize(직렬화, pickling), Deserialize(역직렬화, unpickiling)을 도와주는 데이터 타입
##### ●  Serialize : 객체(list, dict 등) 타입 -> 파일 저장
```python
import pickle

arr = [1, 100, 'A', 3.141592]   # list 객체 타입
with open('array.pickle', mode='wb') as f:     # W: write, b: binary   -> 2진수를 저장하겠다는 의미
    pickle.dump(arr, f)    # 객체(obj)를 파일(f)에 저장
```
##### ●  Deserialize : 파일 -> 객체(list, dict 등) 타입
```python
with open('array.pickle', mode='rb') as f:
    data = pickle.load(f)
print(data)
```

##### ●  연습
```python
data ={
    'name': '오쌤',
    'age': 16,
    'k1': [1, 2.0, 'AB'],
    'k2': {'tel': '010-0000-0000', 'email': 'jake@test.com'}}

# data객체를 data.pkl 파일에 저장(serialization)
with open('data.pkl', mode='wb') as f:
    pickle.dump(data, f)

# data.pkl 파일을 읽어서 dict 객체를 복원(deserialization)
with open('data.pkl', mode='rb') as f:
    data_pkl = pickle.load(f)
print(data_pkl)
```

***********************************************************
## ex08
### ※ MNISt 숫자 손글씨 데이터 신경망 구현
#### 1) 가중치, 편향 행렬 생성
```python
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
```

#### 2) 가중치, 편향을 전달받아서 테스트 데이터의 예측값 예측
```python
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
```

#### 3) 테스트 데이터의 레이블과 테스트 데이터의 예측값을 통해 정확도 계산
```python
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
```

```python
if __name__ == '__main__':
    network = init_network()

    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    pred = predict(network, X_test)
    print('max:',np.max(pred[0]))
    print('y_test[0]:', y_test[0])

    acc = accuracy(y_test, pred)
    print('acc:', acc)
```
###### [2.6744647e-08 8.3487522e-10 2.2669194e-07 3.9877449e-07 3.7157974e-10 1.4254578e-08 5.1546488e-12 3.1590319e-04 2.9701399e-09 2.5926784e-07]
###### max: 0.0003159032
###### y_test[0]: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
###### acc: 0.0001

***********************************************************
## ex09
### ※ PIL 패키지와 numpy 패키지를 이용하여 이미지 파일의 픽셀 정보를 numpy.ndarray형식으로 변환하거나 numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장
```python
import numpy as np
from PIL import Image

def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file, mode='r')    # 이미지 파일 오픈
    print('type: ',type(img))
    pixels = np.array(img)    # 이미지 파일 객체를 numpy.ndarray 형식으로 변환
    print('pixels shape: ', pixels.shape)

    return pixels
```

```python
def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel)   # ndarray 타입의 데이터를 이미지로 변환
    print('type: ',type(img))
    img.show()      # 이미지 뷰어를 사용해서 이미지 보기
    img.save(image_file)   # 이미지 객체를 파일로 저장

```

```python
if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    pixels1 = image_to_pixel('크리스마스.jpg')    # shape: (497, 860, 3)  -> (세로 길이, 가로 길이, color)
    pixels2 = image_to_pixel('크리스마스트리.png')  # shape: (1208, 860, 3)

    pixel_to_image(pixels1, 'test.jpg')
    pixel_to_image(pixels2, 'test2.png')
```

***********************************************************
## ex10 Numpy Broadcast
##### ※ numpy array의 축(ㄴ모양으로 axis0 -> 1) -> pandas도 동일

### array와 scalar간의 Broadcast
#### scalar를 array와 같은 shape으로 만들어서 연산(array의 차원은 상관없음)
```python
x = np.array([1,2,3])
    print('x =', x)
    print('x + 10=', x + 10)
```

#### 2차원 array와 1차원 array간의 Broadcast
##### -> Broadcast가 가능하려면, 2개의 array의 차원이 다른 경우 작은 차원의 array가 큰 차원의 array에 맞춰지게 욉
##### -> but, (2,3)과 (3, )과 같은 모양의 차원이어야 함(작은 차원의 array의 index중 하나가 1이고, 두 array의 column의 수가 같거나 row의 수가 같은 경우   ->  (n , m)와 (m, ))

```python
X = np.arange(6).reshape((2,3))
    a = np.arange(1,4)
    print('X + a \n',X + a)
```

***********************************************************
## ex11 Softmax 함수에 2차원 배열을 사용할 수 있게 수정해서 신경망 
#### 1) 2차원 배열을 사용할 수 있는 softmax함수
```python
import pickle
import numpy as np
from ch03.ex01 import sigmoid
from dataset.mnist import load_mnist


def softmax(X):
    """
    1) X - 1차원: [x_1, x_2, ..., x_n]
    1) X - 2차원: [[x_11, x_12, ..., x_1n],
                   [x_21, x_22, ..., x_2n],
                   ...]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음.
        X = X - m  # 0 이하의 숫자로 변환 <- exp함수의 overflow를 방지하기 위해서.
        y = np.exp(X) / np.sum(np.exp(X))
    elif dimension == 2:
        # m = np.max(X, axis=1).reshape((len(X), 1))
        # # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum
        Xt = X.T  # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T
    return y
```

#### 2) forward 신경망 구현 함수 
```python
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    z1 = sigmoid(x.dot(W1)+b1)
    z2 = sigmoid(z1.dot(W2)+b2)
    y = softmax(z2.dot(W3)+b3)

    return y
```

#### 3) mini batch를 통한 forward 신경망 구현 함수 
```python
def mini_batch(network, X, batch_size):
    y_pred = np.array([])    # 예측값들을 저장할 배열
    # batch_size 만큼씩 X의 데이터들을 나눠서 forward propagation(전파)
    for i in range(0, len(X), batch_size):
        X_batch = X[i:(i+batch_size)]
        y_hat = forward(network, X_batch)        # (batch_size, 10) shape의 배열
        predictions = np.argmax(y_hat, axis=1)   # 각 row에서 최대값의 인덱스  -  (batch_size, ) 배열-1차원
        y_pred = np.append(y_pred, predictions)  # 예측값들을 결과 배열에 추가
    return y_pred   # (len(x), ) shape의 배열
```

#### 4) 정확도 계산 
```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

```python
if __name__ == '__main__':
    np.random.seed(2020)
    # 1차원 softmax 테스트
    a = np.random.randint(10, size=5)
    print(a)
    print(softmax(a))

    # 2차원 softmax 테스트
    A = np.random.randint(10, size=(2, 3))
    print(A)
    print(softmax(A))

    # (Train/Test) 데이터 세트 로드.
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print('X_test.shape :', X_test.shape)
    print('y_test.shape :', y_test.shape)
    # print(X_test[0])
    # print(y_test[0])

    # 신경망 생성 (W1, b1, ...)
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    print('network:', network.keys())
    print('W1:', network['W1'].shape)
    print('W2:', network['W2'].shape)
    print('W3:', network['W3'].shape)

    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print('true[:10]', y_test[:10])
    print('pred[:10]', y_pred[:10])
    print('true[-10:]', y_test[-10:])
    print('pred[-10:]', y_pred[-10:])

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print('정확도:', acc)
```


