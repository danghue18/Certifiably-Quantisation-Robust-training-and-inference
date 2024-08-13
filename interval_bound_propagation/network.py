import torch
import torch.nn as nn
import math
import copy
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from module import *
import torch.nn.utils.spectral_norm as SpectralNorm


class MNIST_MLP(nn.Module):
    def __init__(self,
                non_negative = [True, True, True], 
                norm = [False, False, False]):
        
        super(MNIST_MLP, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1, 128, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(128, 128, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear(128, 10, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.activation = RobustReLu()
        self.score_function = self.fc3
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        return x

