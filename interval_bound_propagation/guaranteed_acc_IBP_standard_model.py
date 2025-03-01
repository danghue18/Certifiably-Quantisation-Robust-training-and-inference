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

from network import *

from compute_acc import *



# parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--ep_i', default=2/255, type=float, help='epsilon_input')
# parser.add_argument('--ep_w', default=2/255, type=float, help='epsilon_weight')
# parser.add_argument('--ep_b', default=2/255, type=float, help='epsilon_bias')
# parser.add_argument('--ep_a', default=2/255, type=float, help='epsilon_activation')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
net =  CIFAR10_MLP(
    non_negative = [False, False, False, False, False, False], 
    norm = [False, False, False,False, False, False])

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\RobustQuantization\nn_quantization_pytorch\nn-quantization-pytorch\quantization\qat\mxt-sim\ckpt\mlp_cifar10\2024-12-05_20-13-00\last_checkpoint.pth.tar')
qat_state_dict = checkpoint['state_dict']


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

def IBP_guaranteed_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0):
    loader = 0
    loadertype = ''
    if test:
        loader = testloader
        loadertype = 'test'
    else:
        loader = trainloader
        loadertype = 'train'
    correct = 0
    total = 0
    check_loss = 0
    with torch.no_grad():
        for batch_idx,  (images, labels)  in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)
            x_ub = images + ep_i
            x_lb = images - ep_i
            
            outputs = net(torch.cat([x_ub,x_lb], 0), ep_w, ep_b, ep_a)
            
            z_ub = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            #print(z_lb==z_ub)
            #loss_nor = criterion(z_ub, labels)
            lb_mask = torch.eye(10).to(device)[labels]
            ub_mask = 1 - lb_mask
            outputs = z_lb * lb_mask + z_ub * ub_mask
            loss = criterion(outputs, labels)
            check_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (check_loss/(batch_idx+1), 100.*correct/total, correct, total))

    correct = correct / total
    print('@number of batch: ' ,len(loader))
    loss_aver = check_loss/len(loader)

    print('Accuracy of the network on the', total, loadertype, 'images: ',correct, 'with epsilon input = ', ep_i,' epsilon param = ', ep_w, ' epsilon activation = ', ep_a )
    return correct, loss_aver

def main():

    print('Evaluate robust 8bit qat model with KURT: ')


    print_accuracy(net, trainloader, testloader, device, test=True, 
                                                                                ep_i =0, 
                                                                                ep_w = 0,
                                                                                ep_b = 0,
                                                                                ep_a = 0)
    print_accuracy(net, trainloader, testloader, device, test=True, 
                                                                                ep_i =1/511, 
                                                                                ep_w = 1/511,
                                                                                ep_b = 1/511,
                                                                                ep_a = 1/511)

if __name__=="__main__":
    main()