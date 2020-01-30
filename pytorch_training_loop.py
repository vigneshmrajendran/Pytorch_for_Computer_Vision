
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import functools
# torch.set_grad_enabled(True)  #optional
# torch.set_printoptions(precision=4, linewidth=120)

training_set = torchvision.datasets.FashionMNIST(
  root = './data/FashionMNIST',
  train=True,
  download=True,
  transform = transforms.Compose([
    transforms.ToTensor()
  ])
)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.fc3 = torch.nn.Linear(in_features=60, out_features=10)

        #Operations
        self.maxpool2d2k2s = functools.partial(F.max_pool2d, kernel_size=2, stride=2)
        self.relu = F.relu
        self.softmax = F.softmax

    def forward(self, t):
        t = self.conv1(t)
        t = self.relu(t)
        t = self.maxpool2d2k2s(t)

        t = self.conv2(t)
        t = self.relu(t)
        t = self.maxpool2d2k2s(t)

        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = self.relu(t)

        t = self.fc2(t)
        t = self.relu(t)

        t = self.fc3(t)
        return t

def get_num_correct(predictions, labels):
    return torch.argmax(predictions, dim=1).eq(labels).sum().item()

new_network = Network()
data_loader = torch.utils.data.DataLoader(training_set, batch_size=100)
optimizer = optim.Adam(new_network.parameters(), lr=0.01)
NUM_EPOCHS = 10

for i in range(NUM_EPOCHS):
    total_loss = 0
    total_correct = 0
    loss_list = []
    for batch in data_loader:
        
        images, labels = batch

        predictions = new_network(images)
        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_correct += get_num_correct(predictions, labels)
        total_loss += loss.item()
        loss_list.append(loss.item())

    print("epoch: ", i,' loss: ', total_loss, ' correct: ', total_correct)