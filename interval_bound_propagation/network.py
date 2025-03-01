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
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        #x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds

class MNIST_6layers(nn.Module):
    def __init__(self,
                non_negative = [False, False, False, False, False, False], 
                norm = [False, False, False,False, False, False],
                n_hidden_nodes=512):
        
        super(MNIST_6layers, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1,  n_hidden_nodes, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(n_hidden_nodes, n_hidden_nodes, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.fc5 = RobustLinear(n_hidden_nodes, n_hidden_nodes, non_negative=non_negative[4])
        if norm[4]:
            self.fc5 = SpectralNorm(self.fc5)

        self.fc6 = RobustLinear(n_hidden_nodes, 10, non_negative=non_negative[5])
        if norm[5]:
            self.fc6 = SpectralNorm(self.fc6)
        self.activation = RobustReLu()
     
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc6(x, epsilon_w, epsilon_b)
        return x
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        #x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc4(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc5(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc6(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds

class MNIST_5layers(nn.Module):
    def __init__(self,
                non_negative = [False, False, False, False, False], 
                norm = [False, False, False,False, False],
                n_hidden_nodes=512):
        
        super(MNIST_5layers, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1,  n_hidden_nodes, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(n_hidden_nodes, n_hidden_nodes, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.fc5 = RobustLinear(n_hidden_nodes, 10, non_negative=non_negative[4])
        if norm[4]:
            self.fc5 = SpectralNorm(self.fc5)
        self.activation = RobustReLu()
     
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc5(x, epsilon_w, epsilon_b)
        return x
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        #x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc4(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc5(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds

class MNIST_4layers(nn.Module):
    def __init__(self,
                non_negative = [False, False, False, False], 
                norm = [False, False, False,False],
                n_hidden_nodes=512):
        
        super(MNIST_4layers, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1,  n_hidden_nodes, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(n_hidden_nodes, 10, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.activation = RobustReLu()
     
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        return x
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        #x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc4(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds

class MNIST_3layers(nn.Module):
    def __init__(self,
                non_negative = [False, False, False], 
                norm = [False, False, False],
                n_hidden_nodes=512):
        
        super(MNIST_3layers, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1,  n_hidden_nodes, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear( n_hidden_nodes,  n_hidden_nodes, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear( n_hidden_nodes,  10, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.activation = RobustReLu()
     
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        return x
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        #x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds

class FMNIST_MLP(nn.Module):
    def __init__(self,
                non_negative = [True, True, True, True, True, True], 
                norm = [False, False, False,False, False, False ]):
        
        super(FMNIST_MLP, self).__init__()
        self.fc1 = RobustLinear(28 * 28 * 1, 256, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(256, 128, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear(128, 64, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(64,32, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.fc5 = RobustLinear(32, 16, non_negative=non_negative[4])
        if norm[4]:
            self.fc5 = SpectralNorm(self.fc5)

        self.fc6 = RobustLinear(16, 10, non_negative=non_negative[5])
        if norm[5]:
            self.fc6 = SpectralNorm(self.fc6)
        
        self.activation = RobustReLu()
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc6(x, epsilon_w, epsilon_b)
        return x

    def bound_propagation(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc6(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds
    
    def linear_bound(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc2(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc3(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc4(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc5(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.activation(x, epsilon_a)

        x = self.fc6(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds


class CIFAR10_MLP(nn.Module):
    def __init__(self,
                non_negative = [True, True, True, True, True, True], 
                norm = [False, False, False,False, False, False ]):
        
        super(CIFAR10_MLP, self).__init__()
        self.fc1 = RobustLinear(32 * 32 * 3, 256, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(256, 128, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear(128, 64, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(64,32, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.fc5 = RobustLinear(32, 16, non_negative=non_negative[4])
        if norm[4]:
            self.fc5 = SpectralNorm(self.fc5)

        self.fc6 = RobustLinear(16, 10, non_negative=non_negative[5])
        if norm[5]:
            self.fc6 = SpectralNorm(self.fc6)
        
        self.activation = RobustReLu()
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc6(x, epsilon_w, epsilon_b)
        return x

    def bound_propagation(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc6(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds




class SVHN_MLP(nn.Module):
    def __init__(self,
                non_negative = [True, True, True, True, True, True], 
                norm = [False, False, False,False, False, False ]):
        
        super(SVHN_MLP, self).__init__()
        self.fc1 = RobustLinear(32 * 32 * 3, 256, non_negative=non_negative[0])
        if norm[0]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(256, 128, non_negative=non_negative[1])
        if norm[1]:
            self.fc2 = SpectralNorm(self.fc2)

        self.fc3 = RobustLinear(128, 64, non_negative=non_negative[2])
        if norm[2]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.fc4 = RobustLinear(64,32, non_negative=non_negative[3])
        if norm[3]:
            self.fc4 = SpectralNorm(self.fc4)
        
        self.fc5 = RobustLinear(32, 16, non_negative=non_negative[4])
        if norm[4]:
            self.fc5 = SpectralNorm(self.fc5)

        self.fc6 = RobustLinear(16, 10, non_negative=non_negative[5])
        if norm[5]:
            self.fc6 = SpectralNorm(self.fc6)
        
        self.activation = RobustReLu()
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc6(x, epsilon_w, epsilon_b)
        return x

    def bound_propagation(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        bounds = []
        x = x.view(x.shape[0], -1)
        x = self.fc1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc3(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc4(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc5(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        x = self.fc6(x, epsilon_w, epsilon_b)
        x_ub = x[:x.shape[0]//2]
        x_lb = x[x.shape[0]//2:]
        bounds.append((x_ub, x_lb))
        return bounds


class Small_ConvNet(nn.Module):
    def __init__(self,
                 non_negative = [True, True, True, True], 
                 norm = [False, False, False, False]):
        
        super(Small_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(1,16,4,2, padding = 1, non_negative =non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(16,32,4,1, padding= 1, non_negative =non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.fc1 = RobustLinear(13*13*32, 100, non_negative =non_negative[2])
        if norm[2]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(100,10, non_negative =non_negative[3])
        if norm[3]:
            self.fc2 = SpectralNorm(self.fc2)
        
        self.activation = RobustReLu()
        
    def forward(self,x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = self.conv1(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.conv2(x, epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc1(x.view(x.shape[0], -1), epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.fc2(x, epsilon_w, epsilon_b)
        return x

class Cifar_Small_ConvNet(nn.Module):
    def __init__(self,
                 non_negative = [True, True, True, True], 
                 norm = [False, False, False, False]):
        
        super(Cifar_Small_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(3,16,4, stride = 2, padding = 0, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(16,32,4, stride = 1, padding= 0, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.fc1 = RobustLinear(12*12*32, 100, non_negative = non_negative[2])
        if norm[2]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(100,10, non_negative = non_negative[3])
        if norm[3]:
            self.fc2 = SpectralNorm(self.fc2)
            
        # self.deconv1 = nn.ConvTranspose2d(32,16,4, padding = 0, stride = 1)
        # self.deconv2 = nn.ConvTranspose2d(16,3,4, padding = 0, stride = 2)
        
        self.activation = RobustReLu()
        #self.score_function = self.fc2
        
        #self.image_norm = ImageNorm([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    
    def forward_conv(self,x,epsilon_w=0, epsilon_b=0, epsilon_a=0):
        #x = self.image_norm(x)
        x = self.conv1(x,epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        x = self.conv2(x,epsilon_w, epsilon_b)
        x = self.activation(x, epsilon_a)
        return x

    def forward_g(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = self.forward_conv(x,epsilon_w, epsilon_b, epsilon_a) 
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x, epsilon_a)
        return x
        
    def forward(self, x, epsilon_w=0, epsilon_b=0, epsilon_a=0):
        x = self.fc2(self.forward_g(x, epsilon_w, epsilon_b, epsilon_a))
        return x  
    

class CIFAR10_MLP_s(nn.Module):
    def __init__(self):
        super(CIFAR10_MLP_s, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
        self.activation = nn.ReLU()  
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x

def MLP_cifar10(): 
    model = CIFAR10_MLP_s()
    return model 

