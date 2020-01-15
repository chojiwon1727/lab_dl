"""
2층 신경망
"""
import numpy as np

from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        weight_init_std를 사용하는 이유 : W1, W2 행렬의 정규분포 첨도를 높여주는 것
        (많은 값들이 중앙에 분포할 수 있도록   ->   대부분의 값들이 비슷비슷하게 만들어서 초기 W 행렬 값들이 유사하게)
        """
        np.random.seed(1231)
        # self.W1 = np.random.randn(784, 32)
        # self.W2 = np.random.randn(32,10)
        # self.b1 = np.zeros(32)
        # self.b2 = np.zeros(10)

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def sigmoid(self, x):
        """
        공식 : 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        공식 : exp(x) / sum(exp(x))
        """
        if x.ndim == 1:
            m = np.max(x)
            x -= m     # overflow 방지
            y = np.exp(x) / np.sum(np.exp(x))

        elif x.ndim == 2:
            xt = x.T
            m = np.max(xt, axis=0)
            xt -= m
            y = np.exp(xt) / np.sum(np.exp(xt), axis=0)
            y = y.T

        return y

    def predict(self, x):
        """
        data -> hidden layer -> output layer -> prediction
        """
        W1, W2= self.params['W1'], self.params['W2']
        b1, b2= self.params['b1'], self.params['b2']

        z1 = self.sigmoid(x.dot(W1) + b1)
        y_pred = self.softmax(z1.dot(W2) + b2)

        return y_pred

    def accuracy(self, x, y_true):
        """

        :param x: 예측값을 구하고 싶은 2차원 배열
        :param y_true: 실제 label(one-hot-encoding되어 있는 2차원 배열)
        :return: 정확도(scalar)
        """
        y_pred = self.predict(x)
        max_pred = np.argmax(y_pred, axis=1)
        max_true = np.argmax(y_true, axis=1)
        acc = np.mean(max_pred == max_true)
        return acc

    def cross_entropy(self, y_true, y_pred):
        """
        공식 : -sum(y_true * log(y_pred + delta))
        """
        if y_pred.ndim == 1:   # 1차원 배열인 경우, 행의 개수가 1개인 2차원 배열로 변환
           y_pred = y_pred.reshape((1, y_pred.size))
           y_true = y_true.reshape((1, y_true.size))

        max_true = np.argmax(y_true, axis=1)        # 정답 index
        n = y_pred.shape[0]                         # 2차원 배열의 row개수
        rows = np.arange(n)                         # [0, 1, 2, 3, ...]   -> row index
        log_p = np.log(y_pred[rows, max_true])      # y_pred[row index, 정답 index]
        entropy = -np.sum(log_p) / n
        return entropy

    def loss(self, x, y_true):
        y_pred = self.predict(x)
        ce = self.cross_entropy(y_true, y_pred)
        return ce

    def numerical_gradient(self, fn, w):
        """
        공식 : lim({f(x+h) - f(x-h)} / {(x+h) - (x-h)})
        """
        # if w.ndim == 1:
        #     gradient = np.zeros_like(w)
        #     delta = 1e-04
        #     for i in range(w.size):
        #         ith_value = w[i]
        #         w[i] = ith_value + delta
        #         fh1 = fn(w)
        #         w[i] = ith_value - delta
        #         fh2 = fn(w)
        #         gradient[i] = (fh1 - fh2) / (2*delta)
        #         w[i] = ith_value
        # elif w.ndim == 2:
        # return gradient

        h = 1e-04
        grad = np.zeros_like(w)
        with np.nditer(w, flags=['c_index', 'multi_index'], op_flags=['readwrite']) as it:
            while not it.finished:
                i = it.multi_index
                ith_value = it[0]
                it[0] = ith_value + h
                fh1 = fn(w)
                it[0] = ith_value - h
                fh2 = fn(w)
                grad[i] = (fh1 - fh2) / (2*h)
                it[0] = ith_value
                it.iternext()
        return grad

    def gradients(self, x, y_true):
        loss_fn = lambda W: self.loss(x, y_true)
        grad = {}           # W1, b1, W2, b2의 gradient를 저장할 dict
        for key in self.params:
            grad[key] = self.numerical_gradient(loss_fn, self.params[key])

        # grad['W1'] = self.numerical_gradient(loss_fn, self.params['W1'])
        # grad['W2'] = self.numerical_gradient(loss_fn, self.params['W2'])
        # grad['b1'] = self.numerical_gradient(loss_fn, self.params['b1'])
        # grad['b2'] = self.numerical_gradient(loss_fn, self.params['b2'])

        return grad


if __name__ == '__main__':
    network = TwoLayerNetwork(784, 32, 10)

    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # y_pred1 = network.predict(X_train[:5])
    # y_true = y_train[:5]
    # print(y_pred1)
    # print(y_train[:5])


    acc = network.accuracy(X_train, y_train)
    print('accuracy:', acc)

    ce = network.loss(X_train[:5], y_train[:5])
    print('ce:', ce)
    ce = network.loss(X_train[:100], y_train[:100])
    print('ce:', ce)

    gradients = network.gradients(X_train[:100], y_train[:100])
    for key in gradients:
        print(key, np.sum(gradients[key]))

    lr = 0.1
    epoch = 1000
    for i in range(epoch):
        for i in range(10):
            gradients = network.gradients(X_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100])
            for i in gradients:
                network.params[key] -= lr * gradients[key]




