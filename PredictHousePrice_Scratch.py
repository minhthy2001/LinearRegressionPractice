import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils import data
from torch import nn

# Reading the data
data = pd.read_csv("data/USA_Housing.csv")
data = data.drop(columns='Address')
data = torch.tensor(data.values, dtype=torch.float32)
data = torch.nn.functional.normalize(data)
print(data)

train = data[0:4001]                                    # Use 80% to train,
test = data[4000:5001]                                  # 20% to test

features = train[:, 0:-1]                               # Separate the labels and features
labels = train[:, -1]
features_test = test[:, 0:-1]
labels_test = test[:, -1]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# Initializing Model Parameters
batch_size = 32
num_examples = 5

indices = list(range(num_examples))
print("List of index: " + str(indices))
random.shuffle(indices)
print("List of index after shuffled: " + str(indices))

batch_indices = torch.tensor(indices[3: min(3 + batch_size, num_examples)])
print("Batch index: " + str(batch_indices))

arr = torch.normal(0, 0.01, size=[5])
print("The arr: " + str(arr))

arr_batch = arr[batch_indices]
print("The batch arr: " + str(arr_batch))


# Defining The Model
# y = X * w + b
def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b


# Defining Loss Function
# loss = (y_hat - Y)^2 / 2
def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# Update Parameters
# W = W - lr * dW/batch_size
# B = B - lr * dW/batch_size
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# Training
w = torch.normal(0, 0.01, size=(features.shape[1], 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01
num_epochs = 10
lossGraph = []

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b)
        loss = squared_loss(y_hat, y)
        # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        loss.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
        lossGraph.append(float(loss.detach().numpy().sum()) / batch_size)
    with torch.no_grad():
        train_loss = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

plt.plot(lossGraph)
plt.show()


# Testing
test_loss = squared_loss(linreg(features_test, w, b), labels_test)
print(f'Loss: {float(test_loss.mean())}')
print(f'Accuracy: {float(100 - 100*test_loss.mean())}')

