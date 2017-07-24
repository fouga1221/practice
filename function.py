import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x =x.T
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return y.T

    x = x - np.max(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if t.size == y.size:
        t = t.argmax(axis = 1)
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def test_entropy(y, t):
    return np.sum((y - t)**2) / 2

def numerical_gradient(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)


