import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


#first using sign function with clamp making value between min value and max value
#sign function's derivative has impulse value(at 0) or 0

class SIGN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.clamp_(-1,1)

# to not make 0, we use np.floor and np.ceil to not make gradients 0
def grad_calculate(self,balance_weight):
    a=grad_number ;;#number of gradient
    for i in range(a):
        if i<np.floor(grad):
            grad[:,:,i]=grad[:,:,i]-balance_weight*(np.floor(grad)-i)
        if i>np.ceil(grad):
            grad[:,:,i]=grad[:,:,i]+balance_weight*(i-np.ceil(grad))





