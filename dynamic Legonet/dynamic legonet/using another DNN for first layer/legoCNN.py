import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
class DCNN(nn.Module):
  def __init__(self,ina,out,kernel_size):
    super(DCNN,self).__init__()
    self.ina=ina
    self.out=out
    self.kernel_size=kernel_size
    self.classifier=nn.Linear(512,10)
    self.conv1=nn.Conv2d(self.ina,self.ina,3,padding=1)
    self.conv2=nn.Conv2d(self.ina,self.out,3,padding=1)
    self.conv3=nn.Conv2d(self.out,self.out,3,padding=1)
    self.batch1=nn.BatchNorm2d(self.ina)
    self.batch2=nn.BatchNorm2d(self.out)
    self.pool=nn.MaxPool2d(2,2)
    self.fc1=nn.Linear(self.out*32*32,84)
    self.fc2=nn.Linear(84,10)
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
      elif isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
  def temp_forward(self,x):
    x=F.leaky_relu(self.batch1(self.conv1(x)))
    x=F.leaky_relu(self.batch2(self.conv2(x)))
    x=F.leaky_relu(self.batch2(self.conv3(x)))
    return x
  def forward(self,x):
    x=F.leaky_relu(self.batch1(self.conv1(x)))
    x=F.leaky_relu(self.batch2(self.conv2(x)))
    x=F.leaky_relu(self.batch2(self.conv3(x)))
    #print(x.size(), self.out)
    x=x.view(-1,self.out*32*32)
    #print(x.size(), self.out)
    x=self.fc1(x)
    #print(x.size(), self.out)
    x=self.fc2(x)
    #print(x.size(), self.out)
    return x
class TempCNN(nn.Module):
  def __init__(self,ina,out,kernel_size):
    super(TempCNN,self).__init__()
    self.model=DCNN(ina,out,kernel_size)
    self.model=self.model.cuda()
    self.criterion2=nn.CrossEntropyLoss()
    self.optimizer2=torch.optim.Adam(self.model.parameters(),lr=0.00005,weight_decay=0.6)
    self.scheduler2=optim.lr_scheduler.CosineAnnealingLR(self.optimizer2,100)
  def make_label(self,y):
    self.label=y

  def forward(self,x):
    self.model.train()
    global t1
    if t1==0:
      for i in range(100):
        #self.schuduler2.step()
        self.optimizer2.step()
        self.scheduler2.step()
        output1=self.model(x)
        loss2=self.criterion2(output1,self.label)
        loss2.backward(retain_graph=True)
        self.optimizer2.step()
        self.optimizer2.zero_grad()
        _,predicted3=output1.max(1)
    return self.model.temp_forward(x)


class LegoCNN(nn.Module):
  def __init__(self,ina,out,kernel_size,split,lego):
    super(LegoCNN,self).__init__()
    self.ina, self.out,self.kernel_size,self.split=ina,out,kernel_size,split
    self.lego_channel=int(ina/split)
    self.lego=int(out*lego)
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego,self.lego_channel,self.kernel_size,self.kernel_size)))
    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
  def forward(self,x):
    self.temp_combination=torch.zeros(self.second_filter_combination.size()).cuda()
    self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
    self.temp_combination.requires_grad=True
    out=0
    for i in range(self.split):
      first_lego=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.first_filter,padding=int(self.kernel_size/2))
      second_kernel=self.second_filter_coefficients[i]*self.temp_combination[i]
      out=out+F.conv2d(first_lego,second_kernel)
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
        layers +=[TempCNN(channel,x,3),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        channel=x
        continue
      if x=='A':
        layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
      else:
        layers +=[LegoCNN(channel,x,3,self.split,self.lego),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        channel=x
    layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    return nn.Sequential(*layers)
  def make_label(self,y):
    self.label=y
    for layer in self.features.children():
      if isinstance(layer,TempCNN):
        layer.make_label(y)
  def STE(self,balance_weight):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.STE(balance_weight)
       


  
