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
from interval_bound_propagation.compute_acc import *

from multiprocessing import freeze_support



# #print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# Training transforms with random crop
transform_train = transforms.Compose([
        transforms.RandomCrop([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
# Testing transforms with center crop (optional, depending on your training procedure)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

trainset = torchvision.datasets.SVHN(root='datasets', split='train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.SVHN(root='datasets', split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)





#load numpy files
model_dictionary = {}
folder_name = 'extracted_params/SVHN/old_1_255/'

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

