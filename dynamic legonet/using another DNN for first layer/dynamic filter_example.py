import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from random import *

m=nn.Conv2d(128,128,3,padding=1)
#x=nn.Linear(128,10)
input=torch.randn(128,128,32,32)
output=m(input)
print(m.weight[0][0][0])
output=m(output)
print(m.weight[0][0][0])
x=nn.Linear(32*32*128,10)
output=output.view(output.size(0),-1)
#out=out.view(out.size(0),-1)
result=x(output)
print(result[0])
