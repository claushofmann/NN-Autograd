from nn import DenseLayer
import graph_creation_api as gc
from functools import reduce
from back_prop import BackProp
import numpy as np
import pandas as pd
import os
import urllib.request


mnist_train_link = 'https://pjreddie.com/media/files/mnist_train.csv'
mnist_test_link = 'https://pjreddie.com/media/files/mnist_test.csv'
data_dir = '.data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.isfile(os.path.join(data_dir, 'mnist_test.csv')) and not os.path.isfile(os.path.join(data_dir, 'mnist_train.csv')):
    print('Downloading mnist_train.csv ...')
    urllib.request.urlretrieve(mnist_train_link, os.path.join(data_dir, 'mnist_train.csv'))
    print('Downloading mnist_test.csv ...')
    urllib.request.urlretrieve(mnist_test_link, os.path.join(data_dir, 'mnist_test.csv'))


train = pd.read_csv(os.path.join(data_dir, 'mnist_train.csv'))
y_train = np.array(train.iloc[:, 0])
X_train = np.array(train.iloc[:, 1:]).astype(np.float)

X_train = X_train / np.max(X_train)

test = pd.read_csv(os.path.join(data_dir, 'mnist_test.csv'))
y_test = np.array(test.iloc[:, 0])
X_test = np.array(test.iloc[:, 1:]).astype(np.float)

X_test = X_test / np.max(X_test)

batch_size = 16
img_dim = 784
learning_rate = 0.01
no_epochs = 10

input = gc.value(np.zeros([batch_size, img_dim]))
ys = gc.value(np.zeros([batch_size], np.int))

layers = [DenseLayer(img_dim, 100), DenseLayer(100, 50), DenseLayer(50, 10)]
gradient_nodes = reduce(lambda a, b: a+b, (layer.get_trainable_values() for layer in layers))

a1 = layers[0](input)
h1 = gc.relu(a1)
a2 = layers[1](h1)
h2 = gc.relu(a2)
a3 = layers[2](h2)
score = gc.softmax_cross_entropy(a3, ys)

backprop = BackProp(score)

for epoch in range(no_epochs):
    losses = []
    accuracies = []
    permutation = np.random.permutation(len(X_train))
    for batch_start in range(0, len(X_train), batch_size):
        if batch_start + batch_size < len(X_train):
            batch_perm = permutation[batch_start:batch_start + batch_size]
            batch_X = X_train[batch_perm]
            batch_y = y_train[batch_perm]

            input.set_value(batch_X)
            ys.set_value(batch_y)

            loss = score.get_value()
            losses.append(loss)

            predictions = np.argmax(a3.get_value(), axis=1)
            accuracy = np.mean(predictions == batch_y)
            accuracies.append(accuracy)

            gradients = backprop.compute_gradients(gradient_nodes)

            # print(score.get_value())

            for node, gradient in zip(gradient_nodes, gradients):
                node.set_value(node.get_value() - learning_rate * gradient)

    print('Avg loss: {}'.format(np.mean(losses)))
    print('Avg acc: {}'.format(np.mean(accuracies)))

print('-' * 38)
print('Testing')

losses = []
accuracies = []

for batch_start in range(0, len(X_test), batch_size):
    if batch_start + batch_size < len(X_test):
        batch_X = X_test[batch_start:batch_start + batch_size]
        batch_y = y_test[batch_start:batch_start + batch_size]

        input.set_value(batch_X)
        ys.set_value(batch_y)

        loss = score.get_value()
        losses.append(loss)

        predictions = np.argmax(a3.get_value(), axis=1)
        accuracy = np.mean(predictions == batch_y)
        accuracies.append(accuracy)

print('Avg loss: {}'.format(np.mean(losses)))
print('Avg acc: {}'.format(np.mean(accuracies)))