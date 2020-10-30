from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import random
import time
import os
import argparse
import logging
import glob
import sys

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
n_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory = True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, pin_memory = True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
check=iter(trainloader)
check.next
for i,(images,labels) in enumerate(trainloader):
  if i==0:
    print(images.shape)
#print(shape(trainloader))
