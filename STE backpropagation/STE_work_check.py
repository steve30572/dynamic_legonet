
from time import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.cuda.is_available()
class STE(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input):
    return torch.sign(input)
  @staticmethod
  def backward(ctx,grad_output):
    return grad_output.clamp_(-1,1)
sign=STE.apply
params=torch.randn(4,requires_grad=True)
output=sign(params)
loss=output.mean()
loss.backward()
print(params)
print(params.grad)






