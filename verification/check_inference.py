from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from multiprocessing import freeze_support

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from interval_bound_propagation.network import *
from interval_bound_propagation.compute_acc import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#MNIST
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

n_hidden_nodes = 64
net =  MNIST_5layers(
    non_negative = [False, False, False, False, False], 
    norm = [False, False, False,False, False], 
    n_hidden_nodes=n_hidden_nodes )
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

#load numpy files
model_dictionary = {}
folder_name = f'extracted_params/MNIST/normal_5_layers_{n_hidden_nodes}/'

name = 'linear_layers'
for i in ['1', '2', '3', '4', '5']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    #print(model_dictionary[name+i+'weight'].size())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
    #print('done with '+name+i)

print('Check model dictionary')

import numpy
def feedforward(x,model_dictionary):
    linear_input = x.view(x.size(0),-1)
    for i in ['1', '2', '3', '4']: 
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
    linear_output = torch.matmul(model_dictionary['linear_layers5'+'weight'],linear_input.transpose(0,1))+model_dictionary['linear_layers5'+'bias'][:, None]
    y = F.softmax(linear_output, dim=0)
    return y

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
