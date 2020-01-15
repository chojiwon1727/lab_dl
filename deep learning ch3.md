# ch03. 신경망
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
### 신경망 모델 코드로 구현하기


