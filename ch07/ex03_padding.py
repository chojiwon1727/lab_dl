import numpy as np


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
