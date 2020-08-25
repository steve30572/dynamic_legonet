import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from random import *
import torch.optim as optim
#import torchvision
import torchvision.transforms as transforms
global t1
t1=0
t2=0
class TempCNN(nn.Module):
  def __init__(self,ina,):
    super(TempCNN,self).__init__()
    self.m=nn.Conv2d(ina,ina,3,padding=1)
    self.k=nn.MaxPool2d(kernel_size=2,stride=2)
  def getweight(self):
    return self.m.weight
  def forward(self,x):
    global t1
    t1=t1+1
      #print(t1)
    #print(x.shape)
    self.input_dim=list(x.size())
    #print(self.input_dim)
      
      #m=m.cuda()
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory = True, num_workers=0)
      #first=nn.Parameter(nn.init.kaiming_normal_(torch.rand(128,3,3,3)))
      #first=first.cuda()
      
      
      #criterion2=nn.CrossEntropyLoss()
      #optimizer2 = torch.optim.SGD(m.parameters(), lr=0.2)
      #scheduler2=optim.lr_scheduler.CosineAnnealingLR(optimizer2,10)
#print(kernel_2[0][0])
      #if t2==0:
      #self.optimizer2.step()
        ###
      #self.scheduler2.step()
        #total=0
        #correct=0
      #global t1
      #print(t1)
        #if t1 !=1:
          #print(t1,t2)
       # print("asdasdsad")
    for i in range(1):
          
            output=self.m(x)
            output=self.m(output)
            output=self.m(output)
            if t1<=4:
              output=self.k(output)
              output=self.m(output)
              output=self.m(output)
              output=self.m(output)
            elif t1<=8:
              output=self.k(output)
              output=self.m(output)
              output=self.m(output)
              output=self.m(output)
            elif t1<=12:
              output=self.k(output)
              output=self.m(output)
              output=self.m(output)
              output=self.m(output)
            #output=self.k(output)
            output=self.m(output)
            output=self.m(output)
            output=self.m(output)
            #output=self.k(output)
          #x=nn.Linear(128,10).cuda()
            #print("here")
            output=output.view(output.size(0),-1)
            #print("here")
            #print(output.size())
            FC=nn.Linear(output.size(1),10).cuda()
            #print("here")
            output=FC(output)
    return output
            #print(output.size())
            #loss2=self.criterion2(output,self.label)
            #print(t1,t2,loss2)
    #print(m.weight[0][0][0])
            #if i!=9:
              #loss2.backward(retain_graph=True)
            #else:
              #loss2.backward(retain_graph=True)
            #self.optimizer2.step()
    #if epoch < 10:
    #  model.STE(1e-4)
            #self.optimizer2.zero_grad()
    #print(m.weight[0][0][0])
          #_, predicted=output.max(1)
          #total += self.label.size(0)
          #correct += predicted.eq(self.label).sum().item()
    #print(total, correct)
  #print("end")
  #print(total, correct)
        #total=0
        #correct=0
        #print("here")
        #print(m.weight.size())
         # self.first_1=nn.Parameter(nn.init.kaiming_normal_(self.m.weight))
          #first_lego=F.conv2d(x,self.first_1,padding=1)
    

class LegoCNN(nn.Module):
  def __init__(self,ina,out,kernel_size,split,lego):
    super(LegoCNN,self).__init__()
    self.ina, self.out,self.kernel_size,self.split=ina,out,kernel_size,split
    self.lego_channel=int(ina/split)
    self.lego=int(out*lego)
    self.lego_2=int(self.lego//2)
    #self.first_filter=((torch.zeros(self.lego,self.lego_channel,self.kernel_size,self.kernel_size))).cuda()
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(2,self.lego_2,self.ina,self.kernel_size,self.kernel_size))).cuda()
    self.lego_lego=nn.Parameter(nn.init.kaiming_normal_(torch.cat((self.first_filter[0],self.first_filter[1]))))
    #self.lego_lego=nn.init.normal_(self.lego_lego)

    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.m=nn.Conv2d(self.ina,self.ina,3,padding=1)
    self.m=self.m.cuda()
    self.model=TempCNN(self.ina)

    self.k=nn.MaxPool2d(kernel_size=2,stride=2)
    self.criterion2=nn.CrossEntropyLoss()
    self.optimizer2=torch.optim.SGD(self.model.parameters(),lr=0.1)
    self.scheduler2=optim.lr_scheduler.CosineAnnealingLR(self.optimizer2,10)
    #self.temp_1=0
  def forward(self,x):
    global temp_1,first_lego, lego_lego
    #print(x.size)
    #print(self.lego_lego.shape)
    with torch.autograd.set_detect_anomaly(False):
      self.temp_combination=torch.zeros(self.second_filter_combination.size()).cuda()
      self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
      self.temp_combination.requires_grad=True
      out=0
      #print(torch.isnan(x).any())
      global t1
      #t1=t1+1
      #print(t1)
    #print(x.shape)
      self.input_dim=list(x.size())
    #print(self.input_dim)
      
      #m=m.cuda()
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory = True, num_workers=0)
      #first=nn.Parameter(nn.init.kaiming_normal_(torch.rand(128,3,3,3)))
      #first=first.cuda()
      
      
      #criterion2=nn.CrossEntropyLoss()
      #optimizer2 = torch.optim.SGD(m.parameters(), lr=0.2)
      #scheduler2=optim.lr_scheduler.CosineAnnealingLR(optimizer2,10)
#print(kernel_2[0][0])
      if t2!=3:
        self.optimizer2.step()
        ###
        self.scheduler2.step()
        total=0
        correct=0
        #global t1
        #print(t1)
        if t1 !=1:
          #print(t1,t2)
       # print("asdasdsad")
          self.model.train()
          for i in range(10):
          
            output=self.model(x)

            #print(output.size())
            loss2=self.criterion2(output,self.label)
            #print(t1,t2,loss2)
    #print(m.weight[0][0][0])
            if t2==0:
              loss2.backward(retain_graph=True)
            
              self.optimizer2.step()
    #if epoch < 10:
    #  model.STE(1e-4)
              self.optimizer2.zero_grad()
    #print(m.weight[0][0][0])
          #_, predicted=output.max(1)
          #total += self.label.size(0)
          #correct += predicted.eq(self.label).sum().item()
    #print(total, correct)
  #print("end")
  #print(total, correct)
        #total=0
        #correct=0
        #print("here")
        #print(m.weight.size())
          self.first_1=nn.Parameter(nn.init.kaiming_normal_(self.model.getweight()))
          first_lego=F.conv2d(x,self.first_1,padding=1)
        #print(first_lego.size())
        #print("here")
          if t1<7:
            self.third_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.out,1,1))).cuda()
            self.third_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.out,1,1))).cuda()
            self.temp_combination2=torch.zeros(self.third_filter_combination.size()).cuda()
            self.temp_combination2.scatter_(2,self.third_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
            self.temp_combination2.requires_grad=True
          else:
            self.third_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.ina,1,1))).cuda()
            self.third_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.ina,1,1))).cuda()
            self.temp_combination2=torch.zeros(self.third_filter_combination.size()).cuda()
            self.temp_combination2.scatter_(2,self.third_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
            self.temp_combination2.requires_grad=True
          second_kernel=self.third_filter_coefficients[0]*self.temp_combination2[0]
      ####
        #print("here")
          
          out +=F.conv2d(first_lego,second_kernel)
        #print("end")
      #####
          return out
    
      
      
      #print(self.lego_channel, lego_lego.shape, self.first_filter.shape)
      #for j in range(self.split):
          
          
        first_lego=F.conv2d(x,self.lego_lego,padding=int(self.kernel_size/2))  #temp_1[:,j*self.lego_channel:(j+1)*self.lego_channel
      #print("asdasd")
          
          
        second_kernel=self.second_filter_coefficients[0]*self.temp_combination[0]
          #print(first_lego.shape,second_kernel.shape)
        out+=F.conv2d(first_lego,second_kernel)
      #print(self.lego_lego.shape)
      #print(out.shape,  x.shape)
      #print("finish")
        return out
      out=0
      if t2==1:
        if t1==1:
          first_lego=F.conv2d(x,self.lego_lego,padding=1)
          second_kernel=self.second_filter_coefficients[0]*self.temp_combination[0]
          out+=F.conv2d(first_lego,second_kernel)
          return out
        else:
          first_lego=F.conv2d(x,self.m.weight,padding=1)
          if t1<7:
            self.third_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.out,1,1))).cuda()
            self.third_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.out,1,1))).cuda()
            self.temp_combination2=torch.zeros(self.third_filter_combination.size()).cuda()
            self.temp_combination2.scatter_(2,self.third_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
            self.temp_combination2.requires_grad=True
          else:
            self.third_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.ina,1,1))).cuda()
            self.third_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.ina,1,1))).cuda()
            self.temp_combination2=torch.zeros(self.third_filter_combination.size()).cuda()
            self.temp_combination2.scatter_(2,self.third_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
            self.temp_combination2.requires_grad=True
          second_kernel=self.third_filter_coefficients[0]*self.temp_combination2[0]
          out +=F.conv2d(first_lego,second_kernel)
          return out


    
    
     
      for i in range(self.split):
        if self.input_dim[2]==2:
          #self.temp_first=nn.Conv2d(self.lego_channel,self.lego,self.kernel_size)
          self.temp_first=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.dynamic,padding=1)
        
        else:
          self.temp_first=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],self.dynamic)
        a=x[:,i*self.lego_channel:(i+1)*self.lego_channel]
        print(self.temp_first[0].shape,second.shape, self.count, self.lego)
       
            

        #print("here?")
        first_lego=F.conv2d(x[:,i*self.lego_channel:(i+1)*self.lego_channel],second,padding=int(self.kernel_size/2))
    
        second_kernel=self.second_filter_coefficients[i]*self.temp_combination[i]

        out+=F.conv2d(first_lego,second_kernel)
   
      return out
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
    self.classifier=nn.Linear(512,classes)   #512
  def forward(self,x):
    out=self.features(x)
    #print(out.shape)
    #print(out.size(0))
    out=out.view(out.size(0),-1)
    out=self.classifier(out)
    return out
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


       


  
