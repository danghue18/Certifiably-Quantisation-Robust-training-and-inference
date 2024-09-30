from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

new_path = "C:/Users/hueda/Documents/Model_robust_weight_perturbation"
sys.path.append(new_path) 

from interval_bound_propagation.network import *
from interval_bound_propagation.utils import progress_bar
from interval_bound_propagation.utils import generate_kappa_schedule_MNIST
from interval_bound_propagation.utils import generate_epsilon_schedule_MNIST
from interval_bound_propagation.compute_acc import *

from multiprocessing import freeze_support


#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
net =  CIFAR10_MLP(
    non_negative = [False, False, False, False, False, False], 
    norm = [False, False, False,False, False, False])
    # non_negative = [True, True, True], 
    # norm = [True, True, True])
    # non_negative = [True, True, True], 
    # norm = [False, False, False])
net = net.to(device)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\CIFAR10\running_eps_0.0019569471624266144.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc_rob']
start_epoch = checkpoint['epoch']

print("Model State Dictionary:")
for param_tensor in net.state_dict():
    print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

folder = 'extracted_params/CIFAR10/'
os.makedirs(folder, exist_ok=True)
# Create directory if it doesn't exist
folder_name = 'extracted_params/CIFAR10/old_1_511/'
os.makedirs(folder_name, exist_ok=True)
name = 'linear_layers'
for i in ['1', '2', '3', '4', '5', '6']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.fc'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.fc'+i+'.bias'].cpu().numpy())
print('done with '+name)

