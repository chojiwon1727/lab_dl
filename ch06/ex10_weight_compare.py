"""
MNIST 데이터를 사용한 초기값과 신경망 성능 비교
"""
from ch06.ex02_sgd import Sgd
from ch06.ex05_1_adam import Adam
from common.multi_layer_net import MultiLayerNet
import numpy as np
import matplotlib.pyplot as plt

# 1. 실험 조건 세팅
from dataset.mnist import load_mnist

weight_init_types = {'std 0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu' }

neural_nets = dict()
train_loses = dict()
for key, value in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                     output_size=10, weight_init_std=value)
    train_loses[key] = []   # 실험하면서 손실값 저장

# 2. MNIST train, test 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
iterators = 2_000
batch_size = 128    # 1번 학습에 사용할 샘플 갯수(미니배치)
optimizer = Sgd(learning_rate=0.01)     # 파라미터 최적화를 할 알고리즘
# optimizer = Adam()     # 파라미터 최적화를 할 알고리즘
# optimizer 변경해가면서 테스트

# 3. 2,000번 반복하면서
np.random.seed(109)
for i in range(iterators):
    # 미니 배치 샘플 랜덤 추출
    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]
    # 테스트 할 신경망 종류마다 반복
    for key, value in neural_nets.items():
        # gradient 계산
        gradient = value.gradient(X_batch, Y_batch)
        # 파라미터(W, b) 업데이트
        optimizer.update(value.params, gradient)
        # 손실(Loss) 계산 -> 리스트 추가
        loss = value.loss(X_batch, Y_batch)
        train_loses[key].append(loss)
    # 손실 100번마다 한번씩 츨력
    if i % 100 == 0:
        print(f'\n===== {i}번째 training =====')
        for key, value in train_loses.items():
            print(key, ':', value[-1])

# 4. x축:반복횟수 / y축:손실 그래프
x = np.arange(iterators)
markers = {'std 0.01': 'o', 'Xavier': 's', 'He': 'D'}
for key, value in train_loses.items():
    plt.plot(x, value, label=key, marker=markers[key])    # value넣을 때 sooth curve 찾아보기
plt.title('weight init compare')
plt.xlabel('iterator')
plt.ylabel('loss')
plt.legend()
plt.show()
