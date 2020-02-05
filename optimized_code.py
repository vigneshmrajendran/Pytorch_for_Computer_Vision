#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torchvision import datasets, transforms

from run_management import RunBuilder
from run_management import RunManager

from models import ConvolutionalNetwork


# read a json settings file to get the hyperparameters
hyperparameters = OrderedDict( 
    lr = [0.01],
    batch_size = [100, 1000],
    shuffle = [True],
    num_workers = [0, 1, 16]
)


runs = RunBuilder.get_runs(hyperparameters)


training_set = torchvision.datasets.FashionMNIST(
  root = './data/FashionMNIST',
  train=True,
  download=True,
  transform = transforms.Compose([
    transforms.ToTensor()
  ])
)


m = RunManager()
NUM_EPOCHS = 1
for run in RunBuilder.get_runs(parameters):
    network = ConvolutionalNetwork()
    loader = torch.utils.data.DataLoader(training_set, 
                                         batch_size=run.batch_size, 
                                         shuffle=run.shuffle, 
                                         num_workers=run.num_workers)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    with m.run_setup(run, network, loader):
        for epoch in range(NUM_EPOCHS):
            with m.epoch_setup():
                for images, labels in loader:
                    
                    predictions = network(images)
                    loss = F.cross_entropy(predictions, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    m.track_loss(loss)
                    m.track_num_correct(predictions, labels)
m.save('results')