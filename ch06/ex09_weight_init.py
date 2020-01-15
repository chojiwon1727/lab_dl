"""
가중치 초기값
    - 신경망의 파라미터인 가중치 행렬(W)을 처음에 어떻게 초기화를 하느냐에 따라서
      신경망의 학습 성능이 달라질 수 있다
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


if __name__ == '__main__':
    # 은닉층(hidden layer)에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-10, 10, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)

    plt.plot(x, y_sig, label='sigmoid')
    plt.plot(x, y_tanh, label='tanh')
    plt.plot(x, y_relu, label='relu')

    plt.axvline(color='gray')
    plt.axhline(color='gray')
    plt.ylim((-1.5, 1.5))
    plt.title('activation functions')
    plt.legend()
    plt.show()

    # 가상의 신경망에서 사용할 테스트 데이터(mini batch) 및 신경망 생성
    np.random.seed(108)
    x = np.random.randn(1000, 100)  # -> randn : 정규분포에 맞는 랜덤 넘버 생성
    # 보통 신경망의 데이터를 보낼 때 단위를 맞추기 위해 정규화를 하기 때문에 정규화된 test 데이터 생성

    node_num = 100  # 은닉층의 노드(뉴런) 개수
    hidden_layer = 5 # 은닉층의 개수
    activations = {} # 데이터가 은닉층을 지났을 때 출력되는 값을 저장

    # 은닉층에서 사용하는 가중치 행렬
    # w = np.random.randn((node_num, node_num))
    # a = x.dot(w)
    # z = sigmoid(a)
    # activations[0] = z

    # 위 코드를 5번 반복해야 하므로 for문을 사용해서 가중치 행렬 생성 및 신경망 통과
    # for i in range(hidden_layer):
    #     if i != 0:
    #         x = activations[i-1]
    #     w = np.random.randn(node_num, node_num)
    #     a = x.dot(w)
    #     z = sigmoid(a)
    #     activations[i] = z

    # for i in range(hidden_layer):
        # w = np.random.randn(node_num, node_num)                           -> N(0, 1)
        # w = np.random.randn(node_num, node_num) * 0.01                    -> N(0, 0.01)
        # w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num)     -> N(0, sqrt(1/n)) -> sigma, tanh
        # w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)     -> N(0, sqrt(2/n)) -> relu
        # a = x.dot(w)
        # x = sigmoid(a)
        # x = tanh(a)
        # x = relu(a)
        # activations[i] = x

    # activations의 key(key 0 = 첫번째 은닉층)별로 값의 그래프 그리기
    # for key, z in activations.items():
    #     plt.subplot(1, len(activations), key+1)     # subplot(nrow, ncol, index)
    #     plt.title(f'{key+1} layer')
    #     plt.hist(z.flatten(), bins=30, range=(-1,1))
    # plt.show()

    weight_init_types = {'std 0.01': 0.01, 'Xavier': np.sqrt(1/node_num), 'He': np.sqrt(2/node_num)}
    input_data = np.random.randn(1000, 100)
    for key, val in weight_init_types.items():
        for i in range(hidden_layer):
            x = input_data
            w = np.random.randn(node_num, node_num) * val
            a = x.dot(w)
            # x= sigmoid(a)
            # x= tanh(a)
            x= relu(a)
            activations[i] = x
        for key, z in activations.items():
            plt.subplot(1, len(activations), key+1)
            plt.title(f'{key+1} layer')
            plt.hist(z.flatten(), bins=30, range=(-1,1))
        plt.show()

