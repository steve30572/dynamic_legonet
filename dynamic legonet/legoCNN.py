import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from random import *

class LegoCNN(nn.Module):
  def __init__(self,ina,out,kernel_size,split,lego):
    super(LegoCNN,self).__init__()
    self.ina, self.out,self.kernel_size,self.split=ina,out,kernel_size,split
    self.lego_channel=int(ina/split)
    self.lego=int(out*lego)
    #self.first_filter=((torch.zeros(self.lego,self.lego_channel,self.kernel_size,self.kernel_size))).cuda()
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)))
    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
  def forward(self,x):
    with torch.autograd.set_detect_anomaly(True):
      self.temp_combination=torch.zeros(self.second_filter_combination.size()).cuda()
      self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
      self.temp_combination.requires_grad=True
      out=0
    #print(x.shape)
      self.input_dim=list(x.size())
    #print(self.input_dim)
      if self.input_dim[2]>2:
        self.dynamic=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego_channel,self.lego_channel,self.input_dim[2]-2,self.input_dim[2]-2))).cuda()
      else:
        self.dynamic=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego_channel,self.lego_channel,2,2))).cuda()
      self.count=self.input_dim[0]//self.lego
      second=nn.Parameter(torch.zeros(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)).cuda()
      #second.requires_grad=True
   # third=nn.Parameter(torch.zeros(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)).cuda()
    ##
    
   # self.first_filter=0
     #yahoo=np.zeros((self.lego,self.lego_channel,self.kernel_size,self.kernel_size))
    
    
     
      for i in range(self.split):
        if self.input_dim[2]==2:
          temp_first=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.dynamic,padding=1)
        #print(temp_first.shape)
        else:
          temp_first=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.dynamic)
        for j in range(self.lego):

          for k in range(self.count):
            #a=0
            #a=a+1
            #print("before")
            second=second+3
            #print("after)")
            #yahoo[j]=yahoo[j]+temp_first[self.count*j+k].cpu().var.detach().numpy()
          #second=torch.from_numpy(yahoo)  
         #second[j] = second[j]+temp_first[self.count*(j)+k]
         #torch.add(second[j],temp_first[self.count*(j)+k])
        #self.first_filter[j] += temp_first[self.count*(j)+k]

        #first_lego=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.first_filter,padding=int(self.kernel_size/2))


        first_lego=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],second,padding=int(self.kernel_size/2))
      #out=out+first_lego
        second_kernel=self.second_filter_coefficients[i]*self.temp_combination[i]
      #out=out+F.conv2d(first_lego,self.second_filter_coefficients[i])
        out+=F.conv2d(first_lego,second_kernel)
    #print("before out")
      return out
  def STE(self,balance_weight):
    self.second_filter_combination.grad=self.temp_combination.grad
    index=self.second_filter_combination.argmax(dim=2).view(-1).cpu().numpy()
    unique, number=np.unique(index,return_counts=True)
    unique2,number2=np.unique(number,return_counts=True)
    max_num=0
    min_num=10000
    for i in range(self.lego):
      compare=(self.split*self.out)/self.lego
      i_count=(index==i).sum().item()
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
    self.split,self.lego,self.classes=split,lego,classes
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
        layers +=[LegoCNN(channel,x,3,self.split,self.lego),
                  #nn.Conv2d(channel,x,3,padding=1),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        #print(channel)
        channel=x
        continue
      if x=='A':
        layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
      else:
        layers +=[LegoCNN(channel,x,3,self.split,self.lego),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]    #True   modify
        channel=x
    layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    return nn.Sequential(*layers)
  def STE(self,balance_weight):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.STE(balance_weight)
example_1=nn.Parameter(nn.init.kaiming_normal_(torch.rand(8,2,3,3)))
example_2=nn.Parameter(nn.init.kaiming_normal_(torch.rand(2,16,8,1,1)))
example_3=nn.Parameter(nn.init.kaiming_normal_(torch.rand(2,16,8,1,1)))
input_example=nn.Parameter(nn.init.kaiming_normal_(torch.rand(128,4,32,32)))
exmaple_4=nn.Parameter(nn.init.kaiming_normal_(torch.rand(16,2,3,3)))
example_5=nn.Parameter(nn.init.kaiming_normal_(torch.rand(16,2,3,3)))
first_lego=F.conv2d(input_example[:,0:2],example_1,padding=2)
access=list(example_1.size())
print(access[0])
second_lego=F.conv2d(first_lego,example_2[0])
ex=torch.cat((example_1,example_1))
print(ex.shape)
#i=randint(0,7)
#print(i)
#print(second_lego.shape)
out=0
for i in range(2):
  lego_1=F.conv2d(input_example[:,i*2:(i+1)*2],example_1,padding=2)
  out=out+F.conv2d(lego_1,example_3[i])
print(out.shape)
final_lego=F.conv2d(input_example[:,0:2],exmaple_4,padding=2)
#print(final_lego.shape)
temp_c=torch.zeros(example_2.size())
temp_c.scatter_(2,example_3.argmax(dim=2,keepdim=True),1)
temp_c.requires_grad=True
#print(temp_c[0][0])
#print(example_3[0][0])
kernel_2=example_2*temp_c
print(kernel_2[0][0])



       


  
