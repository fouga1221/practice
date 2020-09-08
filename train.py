import sys, os
sys.path.append(os.pardir)

import numpy as np
from neural_net import NeuralNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

iters_num = 5000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
iter_per_epoch = max(train_size // batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

model = NeuralNet()

model.add_affine(784, 50)
model.add_active("relu")
model.add_affine(50, 10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    model.fit(x_batch, t_batch, learning_rate)

    loss = model.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = model.accuracy(x_train, t_train)
        test_acc = model.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(i) + "回目")
        print(train_acc, test_acc)

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.show()

