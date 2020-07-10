# IMPORT PACKAGES YOU NEED
from time import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.cuda.is_available()

def custom(nn.Module):
  def __init__(self, class_num, filter_num, num):
     super().__init__():
     self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 1,
                                      out_channels = 32,
                                      kernel_size = 5,
                                      stride = 1,
                                      padding = 2),
                         nn.BatchNorm2d(num_feature = 32),
                         nn.ReLu(),
                         nn.MaxPool2d(kernel_size = 2, stride = 2))
     self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 32,
                                      out_channels = 64,
                                      kernel_size = 5,
                                      stride = 1,
                                      padding = 2),
                         nn.BatchNorm2d(num_features = 64),
                         nn.ReLu(),
                         nn.MaxPool2d(kernel_size = 2, stride = 2))
     self.fc = nn.Linear(out_channels * num * num, class_num)
     self.dropout = nn.Dropout()
  def forward(self,x):
    x=self.layer1(x)
    x=self.layer2(x)
    x=self.fc(x)
    return x
  def copy_grad(self,balance_weight):



