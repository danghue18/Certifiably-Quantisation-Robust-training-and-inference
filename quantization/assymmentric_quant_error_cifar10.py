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
net =  CIFAR10_MLP(
    non_negative = [False, False, False,False, False, False], 
    norm = [False, False, False, False, False, False])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


print('==> load full precision model.')
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\CIFAR10\robust_0.0019569471624266144_0.0019569471624266144_0.0019569471624266144.pth')
net.load_state_dict(checkpoint['net'])



#load numpy files
model_dictionary = {}
model_name = 'CIFAR10/robust_1_511/'
folder_name = 'extracted_params/'+model_name

name = 'linear_layers'
for i in ['1', '2', '3','4','5','6']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    #print(model_dictionary[name+'weight'].size())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
print('done with '+name)

def asymmetric_quantize(x, num_bits, x_min=None, x_max=None, per_channel=False):
    qmin = 0
    qmax = (2 ** num_bits) - 1

    # Calculate per-channel range if specified
    if per_channel:
        x_min = x.min(dim=1, keepdim=True)[0] if x_min is None else x_min
        x_max = x.max(dim=1, keepdim=True)[0] if x_max is None else x_max
    else:
        x_min = x.min().item() if x_min is None else x_min
        x_max = x.max().item() if x_max is None else x_max

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale
    zero_point = torch.clamp(zero_point, qmin, qmax).int()

    x_quantized = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
    return x_quantized, scale, zero_point


def asymmetric_dequantize(x_quantized, scale, zero_point, bias=None):
    dequantized = (x_quantized - zero_point) * scale
    if bias is not None:
        dequantized += bias
    return dequantized



def feedforward_asymmetric(x, model_dictionary, num_bits_weight, num_bits_bias, num_bits_activation):
    x = x.view(x.size(0), -1)

    # Quantize activation
    x_quantized, x_scale, x_zero_point = asymmetric_quantize(x, num_bits_activation)

    for i in ['1', '2', '3', '4', '5']:
        name = 'linear_layers' + i

        # Per-channel quantization
        weight_quantized, w_scale, w_zero_point = asymmetric_quantize(
            model_dictionary[name + 'weight'], num_bits_weight, per_channel=True
        )
        bias_quantized, b_scale, b_zero_point = asymmetric_quantize(
            model_dictionary[name + 'bias'], num_bits_bias
        )

        # Dequantize and apply transformations
        weight = asymmetric_dequantize(weight_quantized, w_scale, w_zero_point)
        bias = asymmetric_dequantize(bias_quantized, b_scale, b_zero_point)
        linear_output = torch.matmul(weight, x_quantized.transpose(0, 1)) + bias[:, None]
        linear_output = torch.relu(linear_output).transpose(0, 1)
        x_quantized, x_scale, x_zero_point = asymmetric_quantize(linear_output, num_bits_activation)

    # Final layer
    name = 'linear_layers6'
    weight_quantized, w_scale, w_zero_point = asymmetric_quantize(
        model_dictionary[name + 'weight'], num_bits_weight, per_channel=True
    )
    bias_quantized, b_scale, b_zero_point = asymmetric_quantize(
        model_dictionary[name + 'bias'], num_bits_bias
    )
    weight = asymmetric_dequantize(weight_quantized, w_scale, w_zero_point)
    bias = asymmetric_dequantize(bias_quantized, b_scale, b_zero_point)
    linear_output = torch.matmul(weight, x_quantized.transpose(0, 1)) + bias[:, None]
    y = F.softmax(linear_output, dim=1)

    return y


def test_asymmetric_quantization(model_dictionary, testloader, num_bits_weight, num_bits_bias, num_bits_activation):
    """
    Evaluate the model with asymmetric quantization.

    Args:
    - model_dictionary (dict): Dictionary of model parameters.
    - testloader (torch.utils.data.DataLoader): Test data loader.
    - num_bits_weight (int): Number of bits for weight quantization.
    - num_bits_bias (int): Number of bits for bias quantization.
    - num_bits_activation (int): Number of bits for activation quantization.

    Returns:
    - float: Test accuracy.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Perform feedforward
            outputs = feedforward_asymmetric(inputs, model_dictionary, num_bits_weight, num_bits_bias, num_bits_activation)

            # Get predictions
            _, predicted = outputs.max(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    return 100. * correct / total, bw, ba

              
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
    all_labels = []
    fx_model_outputs_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            ground_truth += targets.tolist()
            all_labels.append(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            # value_fl_batch = feedforward(inputs,model_dictionary,32,32,32,32)
            # print(value_fl_batch.T) # shape: 10x2
            outputs = net(torch.cat([inputs,inputs], 0))
            value_fl_batch = outputs[:outputs.shape[0]//2]
            value_fl_batch = F.softmax(value_fl_batch, dim=1)
            #print(value_fl_batch)
            fl_model_outputs.append(value_fl_batch) # shape: batch x 10
    all_labels = torch.cat(all_labels)
    all_labels = all_labels.numpy()
    print(all_labels)
    fl_model_outputs = torch.cat(fl_model_outputs).cpu()
    print((np.array(fl_model_outputs)).shape)

    #acc = test(model_dictionary,testloader,32,32,32,32)
    acc,_ = print_accuracy(net, trainloader, testloader, device, test=True)
    print("accuracy fl: ",acc)

    acc_fx_list = []
    for bw in range(2,16):
        #bi = 4
        ba = 8
        print("Ba: ", ba, "Bw: ", bw)
        fx_model_outputs = []
        #fx_model_predicts = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                # maximum value and the index of the maximum value
                value_fx_batch = feedforward_asymmetric(inputs,model_dictionary,bw,bw,ba)
                fx_model_outputs.append(value_fx_batch.T)

        fx_model_outputs = torch.cat(fx_model_outputs).cpu()
        fx_model_outputs_list.append(fx_model_outputs)
        acc_fx,Bw,Ba = test_asymmetric_quantization(model_dictionary, testloader,bw,bw,ba)

        Bw_list.append(Bw)
        Ba_list.append(Ba)
        acc_fx_list.append(acc_fx)
        print("acc = ", acc_fx)

    # Calculate L-inf norm between each fx_model_outputs and fl_model_outputs
    #l2_norms = []
    l_inf = []
    l_inf_list = []


    for fx_model_outputs in fx_model_outputs_list:

        #print((fx_model_outputs - fl_model_outputs).shape) # 10k x 10
        diff = abs(fx_model_outputs - fl_model_outputs) #10k x 10
        worst_case_dis = max(diff[np.arange(diff.shape[0]), all_labels])
        l_inf_list.append(worst_case_dis)


    

    result = {'Bx ': Bx_list, 'Bx ': Bx_list, 'Bb ': Bb_list, 'Ba ': Ba_list, 'Quantize_accuracy': acc_fx_list, 'Quantization_error':l_inf_list }
    path = f'quantization_results/CIFAR10/bound_quantization_error_norm_1_assym_act_8_robust_1_511.xlsx'
    DictExcelSaver.save(result,path)

 


    


