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
#print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

net =  FMNIST_MLP(
    non_negative = [False, False, False,False, False, False], 
    norm = [False, False, False, False, False, False])

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)


# Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\FMNIST\running_eps_7_255.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc_rob']
start_epoch = checkpoint['epoch']

print("Model State Dictionary:")
for param_tensor in net.state_dict():
    print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

folder = 'extracted_params/FMNIST/'
os.makedirs(folder, exist_ok=True)
# Create directory if it doesn't exist
folder_name = 'extracted_params/FMNIST/running_7_255/'
os.makedirs(folder_name, exist_ok=True)
name = 'linear_layers'
for i in ['1', '2', '3', '4', '5', '6']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.fc'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.fc'+i+'.bias'].cpu().numpy())
print('done with '+name)

