from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import random
import time
import os
import argparse
import logging
import glob
import sys
t2=0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
n_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False, pin_memory = True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, pin_memory = True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model=vgg_16_lego('vgg16_lego',1,0.5,10)
model=model.cuda()
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.013)   #SGD 0.1     (    /5   )   #0.08 --> 0.8안됨  0.02->0.8 안됨
#scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,400)

def train(epoch):
  model.train()
  train_loss=0
  correct=0
  total=0
  end=time.time()
  #print("before")
  for batch_idx,(inputs,targets) in enumerate(trainloader):
    #print(batch_idx)
    a=[0,0]
    for i in range(len(targets)):
      if targets[i]<=4:
        a[0]+=1
      else:
        a[1]+=1
    inputs,targets=inputs.cuda(),targets.cuda()
    inputs,targets=Variable(inputs),Variable(targets)
    data_time=time.time()
    model.make_label(a)
    outputs=model(inputs)
    #global t1
    #t1=t1+1
    loss=criterion(outputs,targets)
    #print("before first loss.backward")
    loss.backward(retain_graph=True)
    #model.STE(1e-4)
    optimizer.step()
    #if epoch < 10:
    #  model.STE(1e-4)
    optimizer.zero_grad()
    _, predicted=outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    #print(total, correct)
    train_loss +=loss.item()
    model_time=time.time()
    #print(count_memory(model))
    print(model_time-data_time)
    logging.info('Train Epoch: %d Process: %d Total: %d    Loss: %.06f    Data Time: %.03f s    Model Time: %.03f s    Memory %.03fMB', 
                epoch, batch_idx * len(inputs), len(trainloader.dataset), loss.item(), data_time - end, model_time - data_time, count_memory(model))
    end = time.time()
    #global t1
    #t1=0
    #print("after")
def test(epoch):
  model.eval()
  test_loss=0
  correct=0
  total=0
  #with torch.no_grad():
  for batch_idx,(inputs,targets) in enumerate(testloader):
      a=[0,0]
      for i in range(len(targets)):
        if targets[i]==0 or targets[i]==1 or targets[i]==2 or targets[i]==3 or targets[i]==5:
          a[0]+=1
        else:
          a[1]+=1
      inputs,targets=inputs.cuda(),targets.cuda()
      inputs,targets=Variable(inputs),Variable(targets)
      model.make_label(a)
      outputs=model(inputs)
      loss=criterion(outputs,targets)
      test_loss +=loss.item()
      _, predicted=outputs.max(1)
      total +=targets.size(0)
      correct += predicted.eq(targets).sum().item()
  return correct/total,test_loss/len(testloader)
if __name__ == '__main__':
    sys.setrecursionlimit(3000)
    cudnn.benchmark = True
    torch.cuda.manual_seed(2)
    cudnn.enabled = True
    torch.manual_seed(2)
    
    max_correct = 0
    b_a=0
    b_e=0
    real_start=time.time()
    for epoch in range(50):#400
        #if epoch == 10:
        #    optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad and 'combination' not in n], lr=0.1, momentum = 0.9, weight_decay = 0.0005)
        #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,390)
        ###
        optimizer.step()
        ###
        #scheduler.step()
        train(epoch)
        #global t2
        #t2=1
        start=time.time()
        correct, loss = test(epoch)
        t2=0
        #print("train loss: ",loss)
        #print("accuracy: ",correct,epoch+1)
        if correct > max_correct:
            max_correct = correct
            b_e=epoch
            torch.save(model.state_dict(), 'best_mlp.p')
        print('Epoch %d Correct: %d, Max Correct %d, Loss %.06f', epoch, correct, max_correct, loss)
        finish=time.time()
        print(finish-start)
       
    print("best accuracy:",max_correct, b_e)
    print("time for 50 epoch: ",finish-real_start)
# result: 0.8617 337
