import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import functools

class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.fc3 = torch.nn.Linear(in_features=60, out_features=10)

        #Operations
        self.maxpool = functools.partial(F.max_pool2d, kernel_size=2, stride=2)
        self.relu = F.relu
        self.softmax = F.softmax

    def forward(self, t):
        t = self.conv1(t)
        t = self.relu(t)
        t = self.maxpool(t)

        t = self.conv2(t)
        t = self.relu(t)
        t = self.maxpool(t)

#         t = t.reshape(-1, 12*4*4)
        t = t.flatten(start_dim=1)
        t = self.fc1(t)
        t = self.relu(t)

        t = self.fc2(t)
        t = self.relu(t)

        t = self.fc3(t)
        return t
