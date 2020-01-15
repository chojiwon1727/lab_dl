"""
배치 정규화(Batch Normalization)
    - 신경망의 각 층에 미니 배치를 전달할 때마다 정규화를 실행하도록 강제하는 방법
    - 장점
        1) 학습 속도 개선
        2) 파라미터(W, b) 초기값에 크게 의존하지 않음
        3) 과적합을 억제
"""
# 교재 p213의 그래프 그리기
# batch normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교
from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex05_1_adam import Adam
from common.multi_layer_net_extend import MultiLayerNetExtend
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

np.random.seed(110)


# 배치 정규화를 사용하는 신경망과 사용하지 않는 신경망 만들기
# 실험
#  - mini batch iteration 횟수 변경
#  - weight_init_std 값 변경

bn_neural_net = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                    output_size=10, weight_init_std=0.1, use_batchnorm=True)
neural_net = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                 output_size=10, weight_init_std=0.1, use_batchnorm=False)

(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
# 학습 시간을 줄이기 위해서 학습 데이터의 개수 줄이기
X_train = X_train[:1000]
Y_train = Y_train[:1000]

iterator = 200
batch_size = 128
learning_rate = 0.01
neural_optimizer = Sgd()
bn_neural_optimizer = Sgd()
neural_acc_list = []
bn_neural_acc_list = []

for i in range(iterator):
    mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[mask]
    Y_batch = Y_train[mask]
    for network in (neural_net, bn_neural_net):
        gradient = network.gradient(X_batch, Y_batch)
        if network == neural_net:
            neural_optimizer.update(network.params, gradient)
            neural_acc = neural_net.accuracy(X_batch, Y_batch)
            neural_acc_list.append(neural_acc)
        else:
            bn_neural_optimizer.update(network.params, gradient)
            bn_neural_acc = bn_neural_net.accuracy(X_batch, Y_batch)
            bn_neural_acc_list.append(bn_neural_acc)
    print(f'===== {i}번째 training =====')
    print('Without BatchNorm', neural_acc_list[-1])
    print('BatchNorm', bn_neural_acc_list[-1])

    # gradients = neural_net.gradient(X_batch, Y_batch)
    # neural_optimizer.update(neural_net.params, gradients)
    # neural_acc = neural_net.accuracy(X_batch, Y_batch)
    # neural_acc_list.append(neural_acc)
    #
    # bn_gradients = bn_neural_net.gradient(X_batch, Y_batch)
    # bn_neural_optimizer.update(bn_neural_net.params, bn_gradients)
    # bn_neural_acc = bn_neural_net.accuracy(X_batch, Y_batch)
    # bn_neural_acc_list.append(bn_neural_acc)

    # print(f'===== {i}번째 training =====')
    # print('Without BatchNorm', neural_acc)
    # print('BatchNorm', bn_neural_acc)

# 그래프 그리기
x = np.arange(iterator)
plt.plot(x, neural_acc_list, label='Nueral Net')
plt.plot(x, bn_neural_acc_list, label='Batch Normalization')
plt.title('Training Accuracy')
plt.xlabel('Mini Batch Count')
plt.ylabel('accuracy')
plt.legend()
plt.show()