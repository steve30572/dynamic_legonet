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
    self.lego_2=int(self.lego//2)
    #self.first_filter=((torch.zeros(self.lego,self.lego_channel,self.kernel_size,self.kernel_size))).cuda()
    self.first_filter=nn.Parameter(nn.init.kaiming_normal_(torch.rand(2,self.lego_2,self.ina,self.kernel_size,self.kernel_size))).cuda()
    self.lego_lego_first=nn.Parameter(nn.init.kaiming_normal_(torch.cat((self.first_filter[0],self.first_filter[1]))))
    self.lego_lego_second=nn.Parameter(nn.init.kaiming_normal_(torch.cat((self.first_filter[0],self.first_filter[1]))))
    #self.lego_lego=nn.init.normal_(self.lego_lego)

    self.second_filter_coefficients=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    self.second_filter_combination=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.split,self.out,self.lego,1,1)))
    #self.temp_1=0
  def forward(self,x):
    global temp_1,first_lego, lego_lego
    #print(x.size)
    with torch.autograd.set_detect_anomaly(True):
      self.temp_combination=torch.zeros(self.second_filter_combination.size()).cuda()
      self.temp_combination.scatter_(2,self.second_filter_combination.argmax(dim=2,keepdim=True),1).cuda()
      self.temp_combination.requires_grad=True
      out=0
    #print(x.shape)
      self.input_dim=list(x.size())
      max_filter=0
      print("hi")
      print(self.lego_lego_first.shape, self.lego, self.lego_2)
      print(self.first_filter[0].shape)
      first=self.train_dynamic(self.lego_lego_first,self.split,x)
      print("yes")
      second=self.train_dynamic(self.lego_lego_second,self.split,x)
      max_filter=max(first,second)
    #print(self.input_dim)
    
      #self.lego_lego=nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.lego,self.lego_channel,self.kernel_size,self.kernel_size))).cuda()
      #first=random.randrange(0,2)
      #second=random.randrange(0,2)
      #self.lego_lego=torch.cat((self.first_filter[0],self.first_filter[1]))
      
      #print(self.lego_channel, lego_lego.shape, self.first_filter.shape)
      if max_filter==first:
        first_lego=F.conv2d(x,self.lego_lego_first,padding=1)
      else:
        first_lego=F.conv2d(x,self.lego_lego_second,padding=1)
      second_kernel=self.second_filter_coefficients[0]*self.temp_combination[0]
      out +=F.conv2d(first_lego,second_kernel)
      return out
      for j in range(self.split):
          
          if max_filter==first:
            first_lego=F.conv2d(x[:,j*self.lego_channel:(j+1)*self.lego_channel],self.lego_lego_first,padding=int(self.kernel_size/2))  #temp_1[:,j*self.lego_channel:(j+1)*self.lego_channel
          else:
            first_lego=F.conv2d(x[:,j*self.lego_channel:(j+1)*self.lego_channel],self.lego_lego_second,padding=int(self.kernel_size/2))
          
          
          second_kernel=self.second_filter_coefficients[j]*self.temp_combination[j]
          #print(first_lego.shape,second_kernel.shape)
          out+=F.conv2d(first_lego,second_kernel)
      #print(out.shape)
      #print("finish")
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
  def train_dynamic(self,weight,split,FIFO):
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


    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    n_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory = True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, pin_memory = True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    self.model=dynamic_lego('vgg16_lego',10,weight)
    self.model=self.model.cuda()
    self.criterion=nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(model.parameters(), lr=0.15)# 0.1
    self.scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,400)
    cudnn.benchmark = True
    torch.cuda.manual_seed(2)
    cudnn.enabled = True
    torch.manual_seed(2)
    
    max_correct = 0
    for epoch in range(25):#400
        #if epoch == 10:
        #    optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad and 'combination' not in n], lr=0.1, momentum = 0.9, weight_decay = 0.0005)
        #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,390)
        ###
        self.optimizer.step()
        ###
        self.scheduler.step()
        
        self.train3(epoch,FIFO,split,self.input)
        correct, loss = self.test(epoch,FIFO,split,self.input)
        max_correct=max(correct,max_correct)
    return max_correct


  def train3(self,epoch,FIFO,split,targets):
    #self.model.train()
    train_loss=0
    correct=0
    total=0
    end=time.time()
  #print("before")
    #for batch_idx,(inputs,targets) in enumerate(trainloader):
    #inputs,targets=inputs.cuda(),targets.cuda()
      #self.model.input(targets)
    #print(inputs.shape)
    data_time=time.time()
    print("FIFO")
    outputs=self.model(FIFO)
    print("yoohoo")
    loss=self.criterion(outputs,targets)
    print("here")
    #print("before backward")
    loss.backward()
    print("here")
    #print("after backward")
    #model.STE(1e-4)
    self.optimizer.step()
    print("here")
    #if epoch < 10:
    #  model.STE(1e-4)
    self.optimizer.zero_grad()
    print("here")
    _, predicted=outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    train_loss +=loss.item()
    model_time=time.time()
    #logging.info('Train Epoch: %d Process: %d Total: %d    Loss: %.06f    Data Time: %.03f s    Model Time: %.03f s    Memory %.03fMB', 
     #           epoch, batch_idx * len(inputs), len(trainloader.dataset), loss.item(), data_time - end, model_time - data_time, count_memory(model))
    end = time.time()
    #print("after")
  def test(self,epoch,FIFO,split,targets):
    self.model.eval()
    test_loss=0
    correct=0
    total=0
    with torch.no_grad():
      #for batch_idx,(inputs,targets) in enumerate(testloader):
      #inputs,targets=inputs.cuda(),targets.cuda()
      outputs=model(FIFO)
      loss=criterion(outputs,targets)
      test_loss +=loss.item()
      _, predicted=outputs.max(1)
      total +=targets.size(0)
      correct += predicted.eq(targets).sum().item()
    return correct/total,test_loss/len(testloader)
  def input(self,x):
    self.input=x
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
class dynamic_lego(nn.Module):
  def __init__(self,name,classes,weight):
    super(dynamic_lego,self).__init__()
    self.weight=weight
    self.size1,self.size2,self.size3,self.size4=self.weight.shape
    #self.features=self._make_layers(cfg[name])
    self.classes=classes
    #self.classifier=nn.Linear(32*32*32,classes)
  def forward(self,x):
    #print(self.size2)
    #print(self.weight.shape)
    out=F.conv2d(x,self.weight,padding=1)
    #print(out.shape, "aaaaaa")
    #print(out.size(0))
    chance=out.view(out.size(0),-1)
    #print("is this printed")
    self.classifer=nn.Linear(chance.size(1),10)
    final=self.classifer(chance)
    #print(chance.shape)
    return final
  def _make_layers(self,cfg,weight):
    layers=[]
    channel=3
    
    #for i,x, in enumerate(cfg):
      #if i==0:
        #layers +=[nn.Conv2d(channel,x,3,padding=1),
                  #nn.Conv2d(channel,x,3,padding=1),
                  #nn.BatchNorm2d(x),
                  #nn.ReLU(inplace=True)]
        #print(channel)
        #channel=x
        #continue
      #if x=='A':
        #layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
      #else:
        #layers +=[nn.Conv2d(channel,x,3,padding=1),
         #         nn.BatchNorm2d(x),
          #        nn.ReLU(inplace=True)]    #True   modify
        #channel=x
    #layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    return nn.Sequential(*layers)

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
  def input(self,x):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.input(x)
  def STE(self,balance_weight):
    for layer in self.features.children():
      if isinstance(layer,LegoCNN):
        layer.STE(balance_weight)

       


  
