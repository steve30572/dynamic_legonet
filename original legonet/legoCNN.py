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
##we need STE backpropagation function  modified grad_calcuate function
  #in STE backpropagation STE_back.py
  def STE(self,balance_weight):
    self.second_filter_combination.grad=self.temp_combination.grad
    index=self.second_filter_combination.argmax(dim=2).view(-1).numpy()
    unique, number=np.unique(index,return_counts=True)
    unique2,number2=np.unique(number,return_counts=True)
    max_num=0
    min_num=10000
    for i in range(self.lego):
      compare=(self.split*self.out)/self.lego
      i_count=(indx==i).sum().item()
      max_count=max(max_num,i_count)
      min_count=min(min_num,i_count)
      if i_count<np.floor(compare):
        self.second_filter_combination.grad[:,:,i]=self.second_filter_combination.grad[:,:,i]-balance_weight*(np.floor(compare)-i_count)
      elif i_count>np.ceil(compare):
        self.second_filter_combination.grad[:,:,i]=self.second_filter_combination.grad[:,:,i]+balance_weight*(np.floor(compare)-i_count)





cfg={
    'vgg16_lego':[64,64,64,'A',128,128,128,'A',256,256,256,'A',512,512,512,'A',512,512,512,'A'],     
}

class vgg_16_lego(nn.Module):
  def __init__(self,name,split,lego,classes):
    super(vgg_16_lego,self).__init__()
    self.split,self.lego.self.classes=split,lego,classes
    self.features=self._make_layers(cfg[name])
    self.classifier=nn.Linear(512,classes)
  def forward(self,x):
    out=self.features(x)
    out=out.view(out.size(0),-1)
    out=self.classifier(out)
    return out
  def _make_layers(self,cfg):
    layers=[]
    channel=3
    for i,x, in enumerate(cfg):
      if i==0:
        layers +=[nn.Conv2d(channel,x,3,padding=1),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        channel=x
        continue
      if x=='A':
        layers +=[nn.Maxpool2d(kernel_size=2,stride=2)]
      else:
        layers +=[nn.LegoCNN(channel,x,3,self.split,self.lego),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        channel=x
    layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    return nn.Sequential(*layers)
  def STE(self,balance_weight):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.STE(balance_weight)
       

       
  

