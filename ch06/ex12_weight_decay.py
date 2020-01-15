"""
과적합(overfitting): 모델이 학습 데이터는 정확하게 예측을 하지만 학습되지 않은 데이터에 대해서는 정확도가 떨어지는 현상
- 과적합 발생 경우
    1) 학습 데이터가 적은 경우
    2) 파라미터가 너무 많아서 표현력(representational power)이 너무 높은 모델
- 과적합이 되지 않도록 학습하는 방법
    1) regularization: L1, L2-regularization(정칙화, 규제)
    2) dropout
"""
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

neural_net = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                           output_size=10, weight_decay_lambda=0)  # weight_decay_lambda=0 : 가중치 감소를 사용하지 않음

# overfitting을 만들기 위해 데이터의 수를 줄임
X_train = X_train[:300]
Y_train = Y_train[:300]

epochs = 200            # 1epoch : 모든 학습 데이터가 1번씩 학습된 경우
mini_batch_size = 100   # 1번 forward에 보낼 데이터 샘플 개수

# 학습하면서 테스트 데이터의 정확도를 각 에포크마다 기록
train_accuracies = []
test_accuracies = []

optimizer = Sgd(learning_rate=0.01)  # optimizer

for epoch in range(epochs):
    # indices = np.arange(train_size)
    # np.random.shuffle(indices)
    for i in range(iter_per_epoch):
        x_batch = X_train[(i * mini_batch_size):((i+1) * mini_batch_size)]
        y_batch = Y_train[(i * mini_batch_size):((i+1) * mini_batch_size)]
        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracies.append(train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    test_accuracies.append(test_acc)
    print(f'epoch #{epoch}: train={train_acc}, test={test_acc}')

x = np.arange(epochs)
plt.plot(x, train_accuracies, label='Train')
plt.plot(x, test_accuracies, label='Test')
plt.legend()
plt.title(f'Weight Decay (lambda={wd_rate})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
