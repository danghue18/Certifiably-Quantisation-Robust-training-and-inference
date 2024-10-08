from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn

import os
import torchvision.transforms as transforms

import sys

new_path = "C:/Users/hueda/Documents/Model_robust_weight_perturbation"
sys.path.append(new_path) 

from interval_bound_propagation.network import *
from interval_bound_propagation.compute_acc import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_hidden_nodes = 256
net =  MNIST_4layers(
    non_negative = [False, False, False, False], 
    norm = [False, False, False, False], 
    n_hidden_nodes=n_hidden_nodes )
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)
# for param_tensor in net.to('cpu').state_dict():
#     print(f"{param_tensor}\t{net.to('cpu').state_dict()[param_tensor].size()}")
    #print(param_tensor)
# Load checkpoint.
checkpoint_path = f'C:/Users/hueda/Documents/Model_robust_weight_perturbation/interval_bound_propagation/checkpoint/MNIST/robust_4_layers_{n_hidden_nodes}_16bits.pth'
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])

# print("Model State Dictionary:")
# for param_tensor in net.state_dict():
#     print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")
#     #print(param_tensor)


folder = 'extracted_params/MNIST/'
os.makedirs(folder, exist_ok=True)
# Create directory if it doesn't exist
folder_name = f'extracted_params/MNIST/robust_4_layers_{n_hidden_nodes}_16bits/'
os.makedirs(folder_name, exist_ok=True)
name = 'linear_layers'
for i in ['1', '2', '3','4']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.fc'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.fc'+i+'.bias'].cpu().numpy())
print('done with '+name)