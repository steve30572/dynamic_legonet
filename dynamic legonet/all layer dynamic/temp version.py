import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
t1=0
class dynamic(nn.Module):
  def __init__(self,weight,ina,out):
    super(dynamic,self).__init__()
    global t1
    self.weight=weight
    self.CNN=nn.Conv2d(ina,out,1)
    self.weight=self.weight.cuda()
    self.ina=ina
    self.out=out
    size2=1
    if t1<1:
      size2=32
    elif t1<3:
      size2=16
    elif t1<5:
      size2=8
    elif t1<7:
      size2=4
    elif t1<9:
      size2=2
    t1=t1+1
    self.classifier1=nn.Linear(out*size2*size2,200)
    self.classifier2=nn.Linear(200,84)
    self.classifier3=nn.Linear(84,10)
  def forward(self,x):
    #out=F.conv2d(x,self.weight)
    out=self.CNN(x)
    #out=nn.BatchNorm2d(out)
    #out=nn.LeakyReLU(out)
    out=out.view(out.size(0),-1)
    
    out=self.classifier1(out)
    
    out=self.classifier2(out)
    
    out=self.classifier3(out)
    
    return out
  def getweight(self):
    return self.CNN.weight


class LegoCNN(nn.Module):
  def __init__(self,ina,out,kernel_size,split,lego):
    super(LegoCNN,self).__init__()
    self.ina, self.out,self.kernel_size,self.split=ina,out,kernel_size,split
    self.lego_channel=int(ina/split)
    self.lego=int(out*lego)
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)))
    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))   #nn.Parameter
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.model=dynamic(torch.zeros(1,1,1,1),self.lego,self.out)
    self.model=self.model.cuda()
    #self.criterion=nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,400)
    self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.006)
    self.criterion=nn.CrossEntropyLoss()
  def forward(self,x):
    self.temp_combination=torch.zeros(self.second_filter_combination.size()).cuda()
    self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
    self.temp_combination.requires_grad=True
    first_lego=F.conv2d(x,self.first_filter,padding=1)
    temp_lego=first_lego.clone()
    global t2
    
    if t2==0:
     self.model.train()
     for k in range(50):
      outputs=self.model(first_lego)
    #global t1
    #t1=t1+1
      #self.optimizer.step()
      loss2=self.criterion(outputs,self.label)
      self.optimizer.zero_grad()
      #print("asdklajd")
    #print("before first loss.backward")
      loss2.backward(retain_graph=True)
      #print("zxczxc")
    #model.STE(1e-4)
      self.optimizer.step()
    #if epoch < 10:
    #  model.STE(1e-4)
      self.optimizer.zero_grad()
      _, predicted=outputs.max(1)
      total = self.label.size(0)
      correct = predicted.eq(self.label).sum().item()
      #print(total,correct)
      #print(total,correct)
      #train_loss +=loss2.item()
    out=0
    
    self.second_weight=self.model.getweight()
    
    second_kernel=self.second_weight*self.temp_combination[0]
    second_kernel=nn.Parameter(nn.init.kaiming_normal_(second_kernel))
    out=out+F.conv2d(first_lego,second_kernel)
    #for i in range(self.split):
      #first_lego=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.first_filter,padding=int(self.kernel_size/2))
      #second_kernel=self.second_filter_coefficients[i]*self.temp_combination[i]
      #out=out+F.conv2d(first_lego,second_kernel)
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
  def make_label(self, y):
    self.label=y
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
    #'vgg16_lego':[128,128,128,256,256,256,512,512,512,'A','A','A','A'],
    'vgg16_lego':[128,128,'A',128,128,'A',256,256,'A',512,512,'A',512,512,'A'],     
}

class vgg_16_lego(nn.Module):
  def __init__(self,name,split,lego,classes):
    super(vgg_16_lego,self).__init__()
    self.split,self.lego,self.classes=split,lego,classes
    self.features=self._make_layers(cfg[name])
    self.classifier=nn.Linear(512,84)
    self.classifier2=nn.Linear(84,10)   #512
  def forward(self,x):
    x=self.features(x)
    #print(out.shape)
    #print(out.size(0))
    x=x.view(x.size(0),-1)
    x=self.classifier(x)
    x=self.classifier2(x)
    return x
  def make_label(self,y):
    self.label=y
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.make_label(y)
  def _make_layers(self,cfg):
    layers=[]
    channel=3
    for i,x, in enumerate(cfg):
      if i==0:
        layers +=[nn.Conv2d(channel,x,3,padding=1),
                  #nn.Conv2d(channel,x,3,padding=1),
                  nn.BatchNorm2d(x),
                  nn.LeakyReLU(inplace=True)]
        #print(channel)
        channel=x
        continue
      if x=='A':
        layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
      else:
        layers +=[LegoCNN(channel,x,3,self.split,self.lego),
                  nn.BatchNorm2d(x),
                  nn.LeakyReLU(inplace=True)]    #True   modify
        channel=x
    layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    return nn.Sequential(*layers)
  def STE(self,balance_weight):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.STE(balance_weight)
