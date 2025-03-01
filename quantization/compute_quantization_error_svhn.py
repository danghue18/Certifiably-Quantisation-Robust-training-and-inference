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
from interval_bound_propagation.network import *
from interval_bound_propagation.compute_acc import *

from multiprocessing import freeze_support



import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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


# Model
#print('==> Building model..')
net =  SVHN_MLP(
    non_negative = [False, False, False,False, False, False], 
    norm = [False, False, False, False, False, False])
    # non_negative = [True, True, True], 
    # norm = [True, True, True])
    # non_negative = [True, True, True], 
    # norm = [False, False, False])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


print('==> load full precision model.')
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\SVHN\normal.pth')
net.load_state_dict(checkpoint['net'])



#load numpy files
model_dictionary = {}
model_name = 'SVHN/normal/'
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
    quantized_input = quantizeSigned(x,Bi,2)
    for i in ['1', '2','3','4','5']:
        name = 'linear_layers'+i
        quantized_weight = quantizeSigned(model_dictionary[name+'weight'],Bw,torch.from_numpy(np.load('scalars/'+model_name+name+'.weight.npy')).cuda())
        quantized_bias = quantizeSigned(model_dictionary[name+'bias'],Bb,torch.from_numpy(np.load('scalars/'+model_name+name+'.bias.npy')).cuda())
        linear_output = torch.matmul(quantized_weight, quantized_input.transpose(0,1)) + quantized_bias[:, None]
        linear_output =  F.hardtanh_((F.relu(linear_output)).transpose(0, 1), 0, 16)
        quantized_input = quantizeUnsigned(linear_output, Ba,16)

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


# compute quantization errors
if __name__ == '__main__':
    freeze_support()
    fl_model_outputs = []
    ground_truth = []
    fx_model_outputs_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            ground_truth += targets.tolist()
            inputs, targets = inputs.to(device), targets.to(device)

            # value_fl_batch = feedforward(inputs,model_dictionary,32,32,32,32)
            # print(value_fl_batch.T) # shape: 10x2
            outputs = net(torch.cat([inputs,inputs], 0))
            value_fl_batch = outputs[:outputs.shape[0]//2]
            value_fl_batch = F.softmax(value_fl_batch, dim=1)
            #print(value_fl_batch)
            fl_model_outputs.append(value_fl_batch) # shape: batch x 10
        
    fl_model_outputs = torch.cat(fl_model_outputs).cpu()
    print((np.array(fl_model_outputs)).shape)

    #acc = test(model_dictionary,testloader,32,32,32,32)
    acc,_ = print_accuracy(net, trainloader, testloader, device, test=True)
    print("accuracy fl: ",acc)
    acc = test(model_dictionary,testloader,32,32,32,32)
    print("acc fl 32 bit: ", acc)

    acc_fx_list = []
    for bw in range(2,17):
        bi = 8
        ba = 8
        print("Ba: ", ba, "Bw: ", bw)
        fx_model_outputs = []
        #fx_model_predicts = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                # maximum value and the index of the maximum value
                value_fx_batch = feedforward(inputs,model_dictionary,bi,bw,bw,ba)
                fx_model_outputs.append(value_fx_batch.T)

        fx_model_outputs = torch.cat(fx_model_outputs).cpu()
        fx_model_outputs_list.append(fx_model_outputs)
        acc_fx,Bi,Bw,Bb,Ba = test(model_dictionary, testloader,bi,bw,bw,ba)
        Bx_list.append(Bi)
        Bw_list.append(Bw)
        Bb_list.append(Bb)
        Ba_list.append(Ba)
        acc_fx_list.append(acc_fx)
        print("acc = ", acc_fx)

    # Calculate L-inf norm between each fx_model_outputs and fl_model_outputs
    #l2_norms = []
    l_inf = []
    l_inf_list = []


    for fx_model_outputs in fx_model_outputs_list:
        # print(fx_model_outputs.shape)
        # print(fl_model_outputs.shape)
        print((fx_model_outputs - fl_model_outputs).shape) # 10k x 10
        print(torch.norm(fx_model_outputs - fl_model_outputs, p=float('inf'), dim=1).shape) # 10k x 1
        l_inf = torch.max(torch.norm(fx_model_outputs - fl_model_outputs, p=float('inf'), dim=1))
        #l_inf = torch.max(torch.norm(fx_model_outputs - fl_model_outputs, p=1, dim=1))
        l_inf_list.append(float(l_inf))

    

    result = {'Bx ': Bx_list, 'Bx ': Bx_list, 'Bb ': Bb_list, 'Ba ': Ba_list, 'Quantize_accuracy': acc_fx_list, 'Quantization_error':l_inf_list }
    path = f'quantization_results/SVHN/bound_quantization_error_norm_inf_4_in_4_act_normal.xlsx'
    DictExcelSaver.save(result,path)

 





