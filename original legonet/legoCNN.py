import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

class LegoCNN(nn.Module):
  def __init__(self,ina,out,kernel_size,split,lego):
    super(LegoCnn,self).__init__()
    self.ina, self.out,self.kernel_size,self.split=ina,out,kernel_size,split
    self.lego_channel=int(ina/split)
    self.lego=int(out*lego)
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)))
    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
  def forward(self,x):
    self.temp_combination=torch.zeros(self.second_filter_combination.size())
    self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1)
    self.temp_combination.requires_grad=True
    result=0
    for i in range(self.split):
      first_lego=F.conv2d(x[:,i*self.lego_channel,(i+1)*self.lego_channel],self.first_filter,padding=int(self.kernel_size/2))
      second_kernel=self.second_filter_coefficients[i]*self.temp_combination[i]
      out=out+F.conv2d(first_lego,second_kernel)
    return out
  
##we need STE backpropagation function ##
