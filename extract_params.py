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

from models import *
from utils import progress_bar

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/My_PhD/Numerical_Precision/datasets', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/My_PhD/Numerical_Precision/datasets', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


trainset = torchvision.datasets.MNIST(root='\datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=1)

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

classes = ('0','1','2','3','4','5','6','7','8','9')

# Model
print('==> Building model..')
net = mlp()
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# print("Model State Dictionary:")
# for param_tensor in net.state_dict():
#     print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

folder_name = 'extracted_params/'
# Create directory if it doesn't exist
os.makedirs(folder_name, exist_ok=True)
# name = 'conv1.'
# conv_weight = net.state_dict()['module.conv1.weight'].cpu().numpy()
# bn_gamma = net.state_dict()['module.bn1.weight'].cpu().numpy()
# bn_beta = net.state_dict()['module.bn1.bias'].cpu().numpy()
# bn_rm = net.state_dict()['module.bn1.running_mean'].cpu().numpy()
# bn_rv = net.state_dict()['module.bn1.running_var'].cpu().numpy()

# factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
# np.save(folder_name+name+'weight.npy',conv_weight*factor[:,None,None,None])
# np.save(folder_name+name+'bias.npy',bn_beta-bn_rm*factor)
# print('done with '+name)

# for l in ['1','2','3','4']:
#     for s in ['0','1']:
#         for c in ['1','2']:
#             name = 'layer'+l+'.'+s+'.'
#             conv_weight = net.state_dict()['module.'+name+'conv'+c+'.weight'].cpu().numpy()
#             bn_gamma = net.state_dict()['module.'+name+'bn'+c+'.weight'].cpu().numpy()
#             bn_beta = net.state_dict()['module.'+name+'bn'+c+'.bias'].cpu().numpy()
#             bn_rm = net.state_dict()['module.'+name+'bn'+c+'.running_mean'].cpu().numpy()
#             bn_rv = net.state_dict()['module.'+name+'bn'+c+'.running_var'].cpu().numpy()
#             factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
#             np.save(folder_name+name+'conv'+c+'.weight.npy',conv_weight*factor[:,None,None,None])
#             np.save(folder_name+name+'conv'+c+'.bias.npy',bn_beta-bn_rm*factor)
#             print('done with '+name)
#         if (l!='1') and (s=='0'):
#             name = 'layer'+l+'.'+s+'.shortcut.'
#             conv_weight = net.state_dict()['module.'+name+'0.weight'].cpu().numpy()
#             bn_gamma = net.state_dict()['module.'+name+'1.weight'].cpu().numpy()
#             bn_beta = net.state_dict()['module.'+name+'1.bias'].cpu().numpy()
#             bn_rm = net.state_dict()['module.'+name+'1.running_mean'].cpu().numpy()
#             bn_rv = net.state_dict()['module.'+name+'1.running_var'].cpu().numpy()
#             factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
#             np.save(folder_name+name+'weight.npy',conv_weight*factor[:,None,None,None])
#             np.save(folder_name+name+'bias.npy',bn_beta-bn_rm*factor)
#             print('done with '+name)
name = 'linear_layers'
for i in ['1', '3', '5']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.layers.'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.layers.'+i+'.bias'].cpu().numpy())
print('done with '+name)

