import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision

class RecursiveLegoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_split, n_lego, n_layers):
        super(RecursiveLegoConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.n_split = in_channels, out_channels, kernel_size, n_split
        self.basic_channels = in_channels // self.n_split
        self.n_lego = int(self.out_channels * n_lego)
        self.index=int(self.out_channels/self.in_channels)
        self.batchnorm=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        # self.n_lego
        self.lego = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_lego, self.basic_channels, self.kernel_size, self.kernel_size)))

        self.lego_conv2d_list = []
        for i in n_layers:
            if i == 0:
                lego_conv2d_list.append(LegoConv2d(in_channels, out_channels, kernel_size, n_split, self.n_lego))
            else:
                lego_conv2d_list.append(LegoConv2d(out_channels, out_channels, kernel_size, n_split/self.index, self.n_lego))

    def forward(self, x):
        for lego_conv2d in self.lego_conv2d_list:
            x = lego_conv2d(x, self.lego)
            x=self.batchnorm(x)
            x=self.relu(x)
            
        return x

class LegoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_split, n_lego):
        super(LegoConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.n_split = in_channels, out_channels, kernel_size, n_split
        self.basic_channels = in_channels // self.n_split
        self.n_lego = int(self.out_channels * n_lego)
        self.lego = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_lego, self.basic_channels, self.kernel_size, self.kernel_size)))
        self.aux_coefficients = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))
        self.aux_combination = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))
        
    def forward(self, x, lego):
        self.proxy_combination = torch.zeros(self.aux_combination.size()).to(self.aux_combination.device)
        self.proxy_combination.scatter_(2, self.aux_combination.argmax(dim = 2, keepdim = True), 1); self.proxy_combination.requires_grad = True
        
        out = 0
        for i in range(self.n_split):
            lego_feature =  F.conv2d(x[:, i*self.basic_channels: (i+1)*self.basic_channels], lego, padding = self.kernel_size // 2)
            kernel = self.aux_coefficients[i] * self.proxy_combination[i]
            out = out + F.conv2d(lego_feature, kernel)
        return out

    def copy_grad(self, balance_weight):
        self.aux_combination.grad = self.proxy_combination.grad
        # balance loss
        idxs = self.aux_combination.argmax(dim = 2).view(-1).cpu().numpy()
        unique, count = np.unique(idxs, return_counts = True)
        unique, count = np.unique(count, return_counts = True)
        avg_freq = (self.n_split * self.out_channels ) / self.n_lego
        max_freq = 0
        min_freq = 100
        for i in range(self.n_lego):
            i_freq = (idxs == i).sum().item()
            max_freq = max(max_freq, i_freq)
            min_freq = min(min_freq, i_freq)
            if i_freq >= np.floor(avg_freq) and i_freq <= np.ceil(avg_freq):
                continue
            if i_freq < np.floor(avg_freq):
                self.aux_combination.grad[:, :, i] = self.aux_combination.grad[:, :, i] - balance_weight * (np.floor(avg_freq) - i_freq)
            if i_freq > np.ceil(avg_freq):
                self.aux_combination.grad[:, :, i] = self.aux_combination.grad[:, :, i] + balance_weight * (i_freq - np.ceil(avg_freq))
