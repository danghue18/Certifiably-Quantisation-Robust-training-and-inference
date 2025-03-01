from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys

import torchvision
import torchvision.transforms as transforms

import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from interval_bound_propagation.utils import progress_bar
from interval_bound_propagation.utils import DictExcelSaver

from multiprocessing import freeze_support



import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.FashionMNIST(root='datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



#load numpy files
model_dictionary = {}
model_name = 'FMNIST/running_3_255/'
folder_name = 'extracted_params/'+model_name

name = 'linear_layers'
for i in ['1', '2', '3','4','5','6']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    #print(model_dictionary[name+'weight'].size())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
print('done with '+name)

def quantizeSigned(X,B,R=1.0):
    S=1.0/R
    return R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),1.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))
def quantizeUnsigned(X,B,R=2.0):
    S = 2.0/R
    return 0.5*R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),2.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))

def feedforward(x,model_dictionary,Bi,Bw,Bb,Ba):
    x = x.view(x.size(0),-1)
    quantized_input = quantizeSigned(x,Bi,1)
    for i in ['1', '2','3','4','5']:
        name = 'linear_layers'+i
        quantized_weight = quantizeSigned(model_dictionary[name+'weight'],Bw,torch.from_numpy(np.load('scalars/'+model_name+name+'.weight.npy')).cuda())
        quantized_bias = quantizeSigned(model_dictionary[name+'bias'],Bb,torch.from_numpy(np.load('scalars/'+model_name+name+'.bias.npy')).cuda())
        linear_output = torch.matmul(quantized_weight, quantized_input.transpose(0,1)) + quantized_bias[:, None]
        linear_output =  F.hardtanh_((F.relu(linear_output)).transpose(0, 1), 0, 8)
        quantized_input = quantizeUnsigned(linear_output, Ba,8)

    #output layer 

    name = 'linear_layers6'
    quantized_weight = quantizeSigned(model_dictionary[name+'weight'],Bw,torch.from_numpy(np.load('scalars/'+model_name+name+'.weight.npy')).cuda())
    quantized_bias = quantizeSigned(model_dictionary[name+'bias'],Bb,torch.from_numpy(np.load('scalars/'+model_name+name+'.bias.npy')).cuda())
    linear_output = torch.matmul(quantized_weight, quantized_input.transpose(0,1)) + quantized_bias[:, None]
    y = F.softmax(linear_output, dim=0)
    return y


def test(model_dictionary,testloader,Bi,Bw,Bb,Ba):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,predicted = feedforward(inputs,model_dictionary,Bi,Bw,Bb,Ba).max(0)
            total += targets.size(0)
            correct += predicted.eq(targets ).sum().item()
            #print("output: ", feedforward(inputs,model_dictionary,Bw,Ba))

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                    %(100.*correct/total, correct, total))
    return correct/total,Bi,Bw,Bb,Ba               
max_quantize_acc = 0
Bw_opt = 0
Ba_opt = 0
acc_list = []
Bx_list = []
Bw_list = []
Bb_list = []
Ba_list = []


if __name__ == '__main__':
    freeze_support()
    acc,Bi,Bw,Bb,Ba = test(model_dictionary,testloader,32,32,32,32)
   
    for bw in range(2,17):
        ba = 4
        print("Ba: ", ba, "Bw: ", bw)
        acc,Bi,Bw,Bb,Ba = test(model_dictionary,testloader,4,bw,bw,ba)
        acc_list.append(acc)
        Bx_list.append(Bi)
        Bw_list.append(Bw)
        Bb_list.append(Bb)
        Ba_list.append(Ba)

        if(acc > max_quantize_acc):
            max_quantize_acc = acc
            Bw_opt = Bw
            Ba_opt = Ba
    print(max_quantize_acc, Bw_opt,Ba_opt )
    result = {'Bx ': Bx_list, 'Bx ': Bx_list, 'Bb ': Bb_list, 'Ba ': Ba_list, 'Quantize_accuracy': acc_list}
    #path = f'quantization_results/FMNIST/4_in_4_act_robust_acc_1_255.xlsx'
    path = f'quantization_results/FMNIST/new_4_in_4_act_running_3_255.xlsx'
    DictExcelSaver.save(result,path)



 





