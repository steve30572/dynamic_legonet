
from time import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
import torchvision
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






####################2nd way of using STE######################################
#related with grad_calculate at STE_back.py##################################
aux_coefficients=nn.Parameter(init.kaiming_normal_(torch.rand(2,16,8,1,1)))
aux_combination=nn.Parameter(init.kaiming_normal_(torch.rand(2,16,8,1,1)))
proxy_combination=torch.zeros(aux_coefficients.size()).to(aux_combination.device)
#print(aux_coefficients[1][1])
#print(proxy_combination[1][1])
proxy_combination.scatter_(2,aux_combination.argmax(dim=2,keepdim=True),1)
proxy_combination.requires_grad=True
#print(aux_combination.argmax(dim=2,keepdim=True).size())
#print(proxy_combination[1][1])
idxs=aux_combination.argmax(dim=2).view(-1).cpu().numpy()
#print(aux_combination.size())
print(idxs)
unique,count=np.unique(idxs,return_counts=True)
i_freq=(idxs==1).sum().item()
print(i_freq)

 def STE_backward(self, balance_weight):
        aux_combination.grad = proxy_combination.grad
        # balance loss
        idxs = aux_combination.argmax(dim = 2).view(-1).cpu().numpy()
        unique, count = np.unique(idxs, return_counts = True)
        unique, count = np.unique(count, return_counts = True)
        avg_freq = (32 ) / 2
        max_freq = 0
        min_freq = 100
        for i in range(self.n_lego):
            freq = (idxs == i).sum().item()
            max_freq = max(max_freq, freq)
            min_freq = min(min_freq, freq)
            if freq >= np.floor(avg_freq) and freq <= np.ceil(avg_freq):
                continue
            if freq < np.floor(avg_freq):
                aux_combination.grad[:, :, i] = aux_combination.grad[:, :, i] - balance_weight * (np.floor(avg_freq) - freq)
            if freq > np.ceil(avg_freq):
                aux_combination.grad[:, :, i] = aux_combination.grad[:, :, i] + balance_weight * (freq - np.ceil(avg_freq))
