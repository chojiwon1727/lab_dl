"""2층 신경망 테스트"""
import pickle
import numpy as np
from ch05.ex10_twolayer import TwoLayerNetwork
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(106)

    #  MNIST 데이터를 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 2층 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784,
                                 hidden_size=32,
                                 output_size=10)

    # <문제>
    # 100회(epochs)만큼 반복
    # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가
    # 각 epoch마다 테스트 데이터로 테스트를 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.


    # 한번에 학습시키는 입력 데이터 개수(batch_size : 1 -> 한번에 1개씩, 한번 eopchdp 6만번 W, b 변경)
    batch_size = 128
    learning_rate = 0.1
    epochs = 50
    # batch_size를 줄이면 정확도는 높아지지만 시간이 오래 걸림

    # 한번의 epoch(학습이 한번 완료되는 주기)에 필요한 반복 횟수 -> W와 b가 한번의 epoch에서 변경되는 횟수
    iter_size = max(X_train.shape[0] // batch_size, 1)   # 600
    print(iter_size)

    train_losses = []       # 각 epoch마다 학습 데이터의 손실을 저장할 리스트
    train_accuracies = []   # 각 epoch마다 학습 데이터의 정확도을 저장할 리스트
    test_accuracies = []    # 각 epoch마다 테스트 데이터의 정확도을 저장할 리스트

    for epoch in range(epochs):
        # 학습데이터를 랜덤하게 섞음(x_train과 y_train을 같은 순서로 섞어야 함 -> 인덱스 사용)
        idx = np.arange(len(X_train))  # [0, 1, 2, ...., 59999]
        np.random.shuffle(idx)

        for i in range(iter_size):
            # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
            X_batch = X_train[idx[i*batch_size:(i+1)*batch_size]]
            y_batch = Y_train[idx[i*batch_size:(i+1)*batch_size]]
            gradient = neural_net.gradient(X_batch, y_batch)

            # 가중치/편향 행렬들을 수정
            for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradient[key]

        # loss를 계산해서 출력
        train_loss = neural_net.loss(X_train, Y_train)
        train_losses.append(train_loss)
        # print('train_loss :', train_loss)

        # accuracy를 계산해서 출력
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_accuracies.append(train_acc)
        print('train_acc :', train_acc)

        test_acc = neural_net.accuracy(X_test, Y_test)
        test_accuracies.append(test_acc)
        print('test_acc :', test_acc)

    # -> 결과 train acc는 99%지만 test acc는 97%에서 더 이상 증가하지 않는 것을 보면, 학습데이터에 overfitting된 것을 알 수 있음

    # epoch ~ loss 그래프
    x = range(epochs)
    plt.plot(x, train_losses)
    plt.title('Loss - Cross Entropy')
    plt.show()

    # epoch ~ accuracy그래프
    plt.plot(x, train_accuracies, label='train_accuracies')
    plt.plot(x, test_accuracies, label='test_accuracies')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    # 신경망에서 학습이 모두 끝난 후 파라미터(가중치, 편향 행렬)들을 파일에 저장 - pickle
    with open('Weight_bias.pickle', mode='wb') as f:
        pickle.dump(neural_net.params, f)


