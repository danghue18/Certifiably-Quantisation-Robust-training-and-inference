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
from utils import progress_bar
from utils import generate_kappa_schedule_MNIST
from utils import generate_epsilon_schedule_MNIST
from compute_acc import *
from utils import DictExcelSaver

from multiprocessing import freeze_support


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ep_i', default=2/255, type=float, help='epsilon_input')
parser.add_argument('--ep_w', default=2/255, type=float, help='epsilon_weight')
parser.add_argument('--ep_b', default=2/255, type=float, help='epsilon_bias')
parser.add_argument('--ep_a', default=2/255, type=float, help='epsilon_activation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
#print('==> Preparing data..')
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('0','1','2','3','4','5','6','7','8','9')


# Model
#print('==> Building model..')
net =  MNIST_MLP(
    non_negative = [False, False, False], 
    norm = [False, False, False])
    # non_negative = [True, True, True], 
    # norm = [True, True, True])
    # non_negative = [True, True, True], 
    # norm = [False, False, False])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def main():
# Load checkpoint.
    # print('==> evaluate robust model.')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/test_eps_k_0.5/running_eps_2_255.pth')
    # net.load_state_dict(checkpoint['net'])
    # print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 2/255, 
    #                                                                         ep_w = 2/255,
    #                                                                         ep_b = 2/255,
    #                                                                         ep_a = 2/255)

    # print_accuracy(net, trainloader, testloader, device, test=True)
    acc_rob_list = []
    print('Evaluate normal model: ')
    checkpoint_nor = torch.load('./checkpoint/normal/ckpt.pth')
    net.load_state_dict(checkpoint_nor['net'])
    eps_list = [0, 1/255, 2/255, 4/255, 8/255]
    for eps in eps_list: 
        acc_rob, _ = print_accuracy(net, trainloader, testloader, device, test=True, ep_i =eps, 
                                                                                ep_w = eps,
                                                                                ep_b = eps,
                                                                                ep_a = eps)
        acc_rob_list.append(acc_rob)
    result = {'acc_rob': acc_rob_list}
    path = 'results/training_phase/rob_acc_nor_model.xlsx'
    DictExcelSaver.save(result,path)
if __name__=="__main__":
    main()