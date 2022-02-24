import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

# Reading the data
data = pd.read_csv("data/USA_Housing.csv")
data = data.drop(columns='Address')
data = torch.tensor(data.values, dtype=torch.float32)
data = torch.nn.functional.normalize(data)
print(data)

train = data[0:4001]                                    # Use 80% to train,
test = data[4000:5001]                                  # 20% to test

print(train)

features = train[0:4001, 0:5]
labels = train[0:4001, 5:6]

# print(features)
# print(labels)

# define function to load array
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """Construct a PyTorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# neural network with 5-dimension input (5 features) and 1-dimension output (1 label)
net = nn.Sequential(nn.Linear(5, 1))
# initialize model parameters
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss function using Mean Square Error
loss = nn.MSELoss()

# optimization algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# training
num_epochs = 60
lossGraph = []
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    lossGraph.append(float(l.detach().numpy()))
    print(f'epoch {epoch + 1}, loss {l:f}')

plt.plot(lossGraph)
plt.show()

w = net[0].weight.data
b = net[0].bias.data

# testing
features_test = train[4000:5001, 0:5]
labels_test = train[4000:5001, 5:6]

# calculating accuracy
# accuracy can be improved by increase num_epochs or decreased batch_size
y_hat_test = net(features_test)
print("\nLoss:")
loss_test = loss(y_hat_test, labels_test)
print(float(loss_test))
print("\nAccuracy:")
print((1 - loss_test.item()) * 100, "%")
