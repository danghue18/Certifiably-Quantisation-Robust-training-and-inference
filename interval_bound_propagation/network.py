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
