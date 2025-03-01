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



# #print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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



#load numpy files
model_dictionary = {}
folder_name = 'extracted_params/CIFAR10/qat_8bits_no_kure/'

name = 'linear_layers'
for i in ['1', '2', '3', '4', '5', '6']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    #print(model_dictionary[name+i+'weight'].size())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
    #print('done with '+name+i)

print('Check model dictionary')

import numpy
def feedforward(x,model_dictionary):
    linear_input = x.view(x.size(0),-1)
    for i in ['1', '2', '3', '4', '5']: 
        name = 'linear_layers'+i
        # print(linear_input.shape)
        # print(model_dictionary[name+'weight'].T.shape)
        linear_output = torch.matmul((model_dictionary[name+'weight']),linear_input.transpose(0,1))+model_dictionary[name+'bias'][:, None]
        # print(linear_output.size())
        # print(model_dictionary[name+'bias'].size())
        #print(model_dictionary[name+'bias'][:, None].size())
        linear_output = F.relu(linear_output)
        linear_input = linear_output.transpose(0, 1)
        
    #output layer 
    linear_output = torch.matmul(model_dictionary['linear_layers6'+'weight'],linear_input.transpose(0,1))+model_dictionary['linear_layers6'+'bias'][:, None]
    #y = F.softmax(linear_output, dim=0)
    return linear_output

def test(model_dictionary,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.size())
            _,predicted = feedforward(inputs,model_dictionary).max(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                    %(100.*correct/total, correct, total))


if __name__ == '__main__':
    freeze_support()
    test(model_dictionary,testloader)

