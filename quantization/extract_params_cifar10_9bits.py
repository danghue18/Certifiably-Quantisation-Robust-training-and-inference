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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
net =  CIFAR10_MLP_s()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\RobustQuantization\nn_quantization_pytorch\nn-quantization-pytorch\quantization\qat\mxt-sim\ckpt\mlp_cifar10\2025-02-10_15-36-58\model_best.pth.tar')
qat_state_dict = checkpoint['state_dict']


# def dequantize_weights(state_dict, num_bits=8):
#     """Dequantize weights using LSQ alpha values."""
#     dequantized_state_dict = {}
#     qmin = -(2**(num_bits - 1))
#     qmax = 2**(num_bits - 1) - 1
#     for key, value in state_dict.items():
#         if 'weight' in key:
#             # Extract the layer prefix (e.g., "fc6" from "fc6.weight")
#             layer_prefix = key.split('.')[0]
            
#             # Check if the corresponding alpha exists
#             alpha_key = f"{layer_prefix}.alpha"
#             if alpha_key in state_dict:
#                 alpha = state_dict[alpha_key].item()
#                 #dequantized_state_dict[key] = torch.clamp((value/alpha).round(), qmin, qmax)*alpha
#                 dequantized_state_dict[key] = value * alpha
#             else:
#                 # Directly copy weights for layers without alpha
#                 dequantized_state_dict[key] = value
#         else:
#             # Directly copy biases or other parameters
#             dequantized_state_dict[key] = value
    
#     return dequantized_state_dict

# def dequantize_weights(state_dict):
#     """Dequantize weights using LSQ alpha values."""
#     dequantized_state_dict = {}

#     for key, value in state_dict.items():
#         if 'weight' in key:
#             layer_prefix = key.split('.')[0]
#             # Check if the corresponding alpha exists
#             alpha_key = f"{layer_prefix}.alpha"
#             if alpha_key in state_dict:
#                 alpha = state_dict[alpha_key]
#                 if alpha.numel() != 1:
#                     raise ValueError(f"Expected scalar alpha for {alpha_key}, got {alpha.size()}")
#                 alpha = alpha.item()  # Convert to scalar
#                 dequantized_state_dict[key] = value * alpha
#             else:
#                 dequantized_state_dict[key] = value
#         else:
#             dequantized_state_dict[key] = value
    
#     return dequantized_state_dict

def dequantize_weights(state_dict):
    """
    Dequantize weights using stored alpha values from QAT checkpoint.
    """
    dequantized_state_dict = {}
    for key, value in state_dict.items():
        if 'weight' in key:  # Process only weight tensors
            layer_prefix = key.split('.')[0]
            alpha_key = f"{layer_prefix}.alpha"
            if alpha_key in state_dict:  # If alpha exists
                alpha = state_dict[alpha_key].item()
                dequantized_state_dict[key] = value   # Dequantize weight
            else:
                dequantized_state_dict[key] = value  # Copy directly if no alpha
        else:
            dequantized_state_dict[key] = value  # Copy bias and other params
    return dequantized_state_dict

dequantized_state_dict = dequantize_weights(qat_state_dict)
#print(dequantized_state_dict.keys())

filtered_state_dict = {k: v for k, v in dequantized_state_dict.items() if 'alpha' not in k}
filtered_state_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in filtered_state_dict.items()}
net.load_state_dict(filtered_state_dict)

print("Model State Dictionary:")
for param_tensor in net.state_dict():
    print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

folder = 'extracted_params/CIFAR10/'
os.makedirs(folder, exist_ok=True)
# Create directory if it doesn't exist
folder_name = 'extracted_params/CIFAR10/qat_8bits_kure/'
os.makedirs(folder_name, exist_ok=True)
name = 'linear_layers'
for i in ['1', '2', '3', '4', '5', '6']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.fc'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.fc'+i+'.bias'].cpu().numpy())
print('done with '+name)





