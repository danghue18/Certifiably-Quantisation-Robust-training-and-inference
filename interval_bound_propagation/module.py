import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn   
from torch.nn.parameter import Parameter


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1) 
    

class RobustLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, non_negative=True):
        super(RobustLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if non_negative:
            self.weight = Parameter(torch.rand(out_features, in_features) * 1/math.sqrt(in_features))
        else:
            self.weight = Parameter(torch.randn(out_features, in_features) * 1/math.sqrt(in_features))
            
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        self.non_negative = non_negative


    def forward(self, input, epsilon_w=0, epsilon_b=0):
        input_u = input[:input.shape[0]//2]
        input_l = input[input.shape[0]//2:]

        u = (input_u + input_l)/2
        r = (input_u - input_l)/2
        # print(u.shape, self.weight.shape)
        # print((torch.norm(u, p=1, dim=1, keepdim=True)).shape)
        out_u = F.linear(u, self.weight, self.bias) + epsilon_w * torch.norm(u, p=1, dim=1, keepdim=True)
        out_r = F.linear(r, torch.abs(self.weight), None) + epsilon_w * torch.norm(r, p=1, dim=1, keepdim=True) + epsilon_b
        return torch.cat([out_u + out_r, out_u - out_r], 0)
    

class RobustConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, non_negative = True):
        super(RobustConv2d, self).__init__()
        if non_negative:
            self.weight = Parameter(torch.rand(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        else:
            self.weight = Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.non_negative = non_negative

    def forward(self, input, epsilon_w=0, epsilon_b=0):
        input_u = input[:input.shape[0]//2]
        input_l = input[input.shape[0]//2:]

        if self.non_negative:
            out_p = F.conv2d(input_u, F.relu(self.weight+epsilon_w), self.bias+epsilon_b, self.stride,
                        self.padding, self.dilation, self.groups)
            out_n = F.conv2d(input_l, F.relu(self.weight-epsilon_w), self.bias-epsilon_b, self.stride,
                        self.padding, self.dilation, self.groups)
            return torch.cat([out_p, out_n],0)
            
        u = (input_u + input_l)/2
        r = (input_u - input_l)/2 
        # print(u.shape, self.weight.shape)
        # print((torch.norm(u, p=1, dim=1, keepdim=True)).shape)
        out_u = F.conv2d(u, self.weight,self.bias, self.stride,
                        self.padding, self.dilation, self.groups) + epsilon_w * torch.norm(u, p=1, dim=(1,2,3), keepdim=True)
        out_r = F.conv2d(r, torch.abs(self.weight), None, self.stride,
                        self.padding, self.dilation, self.groups) + epsilon_w * torch.norm(r, p=1, dim=(1,2,3), keepdim=True) + epsilon_b
        return torch.cat([out_u + out_r, out_u - out_r], 0)


class RobustReLu(nn.Module):
    def __init__(self):
        super(RobustReLu, self).__init__()

    def forward(self, input, epsilon_a):
        input_u = input[:input.shape[0]//2]
        input_l = input[input.shape[0]//2:]
        
        # Áp dụng ReLU và epsilon_a
        z_U = nn.functional.relu(input_u) + epsilon_a
        z_L = nn.functional.relu(input_l - epsilon_a)
        

        return torch.cat([z_U, z_L], 0)



class ImageNorm(nn.Module):
    def __init__(self, mean, std):
        super(ImageNorm, self).__init__()
        self.mean = torch.from_numpy(np.array(mean)).view(1,3,1,1).cuda().float()
        self.std = torch.from_numpy(np.array(std)).view(1,3,1,1).cuda().float()
        
    def forward(self, input):
        input = torch.clamp(input, 0, 1)
        return (input - self.mean)/self.std
        