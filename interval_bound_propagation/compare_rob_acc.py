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
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='\datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
indices = list(range(0, 1000))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, sampler=indices, num_workers=2)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#classes = ('0','1','2','3','4','5','6','7','8','9')

#Model
print('==> Building model..')
n_hidden_nodes = 256
net =  MNIST_4layers(
    non_negative = [False, False, False,False], 
    norm = [False, False, False, False], 
    n_hidden_nodes=n_hidden_nodes )
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# transform_train = transforms.Compose([
#     #transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     #transforms.Normalize((0.1307,), (0.3081,)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     #transforms.Normalize((0.1307,), (0.3081,)),
# ])

# trainset = torchvision.datasets.FashionMNIST(root='datasets', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

# testset = torchvision.datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# #classes = ('0','1','2','3','4','5','6','7','8','9')


# # Model
# #print('==> Building model..')
# net =  FMNIST_MLP(
#     non_negative = [False, False, False,False, False, False], 
#     norm = [False, False, False, False, False, False])
#     # non_negative = [True, True, True], 
#     # norm = [True, True, True])
#     # non_negative = [True, True, True], 
#     # norm = [False, False, False])
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True



# # Data
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

# trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # Model
# #print('==> Building model..')
# net =  CIFAR10_MLP(
#     non_negative = [False, False, False, False, False, False], 
#     norm = [False, False, False,False, False, False])
#     # non_negative = [True, True, True], 
#     # norm = [True, True, True])
#     # non_negative = [True, True, True], 
#     # norm = [False, False, False])
# net = net.to(device)


# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


#Data
# print('==> Preparing data..')
# # Training transforms with random crop
# transform_train = transforms.Compose([
#         transforms.RandomCrop([32, 32]),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
# # Testing transforms with center crop (optional, depending on your training procedure)
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# trainset = torchvision.datasets.SVHN(root='datasets', split='train', download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# testset = torchvision.datasets.SVHN(root='datasets', split='test', download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # Model
# #print('==> Building model..')
# net =  SVHN_MLP(
#     non_negative = [False, False, False, False, False, False], 
#     norm = [False, False, False,False, False, False])
#     # non_negative = [True, True, True], 
#     # norm = [True, True, True])
#     # non_negative = [True, True, True], 
#     # norm = [False, False, False])
# net = net.to(device)


# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

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

# trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # Model
# #print('==> Building model..')
# net =  CIFAR10_MLP(
#     non_negative = [False, False, False,False, False, False], 
#     norm = [False, False, False, False, False, False])
#     # non_negative = [True, True, True], 
#     # norm = [True, True, True])
#     # non_negative = [True, True, True], 
#     # norm = [False, False, False])
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

def main():
# Load checkpoint.
    # print('==> evaluate robust model.')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\test_eps_k_0.5\fixed_eps_2_255.pth')
    # net.load_state_dict(checkpoint['net'])
    # print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 2/255, 
    #                                                                         ep_w = 2/255,
    #                                                                         ep_b = 2/255,
    #                                                                         ep_a = 2/255)

    # print_accuracy(net, trainloader, testloader, device, test=True)

    acc_rob_input_list = []
    acc_rob_param_list = []
    acc_rob_act_list = []
    acc_rob_input_act_list = []
    acc_rob_input_param_list = []
    acc_rob_param_act_list = []
    acc_rob_list = []
    eps_list = []

    print('Evaluate robust model: ')
    # Load checkpoint.
    checkpoint_path = f'./checkpoint/MNIST/robust_4_layers_256_10bits.pth'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])

    print_accuracy(net, trainloader, testloader, device, test=True, 
                                                                                ep_i =0, 
                                                                                ep_w = 1/512,
                                                                                ep_b = 1/512,
                                                                                ep_a = 0)
    # print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =1/16, 
    #                                                                             ep_w = 1/8,
    #                                                                             ep_b = 1/8,
    #                                                                             ep_a = 1/8)
    # #eps_list = [0, 1/255, 2/255, 3/255, 3/255]
    # step = 0.03/20
    # for i in range(21):
    #     eps = step * i
    #     eps_list.append(eps)
    #     acc_rob_input, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =eps, 
    #                                                                             ep_w = 0,
    #                                                                             ep_b = 0,
    #                                                                             ep_a = 0)
    #     acc_rob_input_list.append(acc_rob_input)
        
    #     acc_rob_param, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =0, 
    #                                                                             ep_w = eps,
    #                                                                             ep_b = eps,
    #                                                                             ep_a = 0)
    #     acc_rob_param_list.append(acc_rob_param)

    #     acc_rob_act, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =0, 
    #                                                                             ep_w = 0,
    #                                                                             ep_b = 0,
    #                                                                             ep_a = eps)
    #     acc_rob_act_list.append(acc_rob_act)  

    #     acc_rob_input_act, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =eps, 
    #                                                                             ep_w = 0,
    #                                                                             ep_b = 0,
    #                                                                             ep_a = eps)
    #     acc_rob_input_act_list.append(acc_rob_input_act) 

    #     acc_rob_param_act, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =0, 
    #                                                                             ep_w = eps,
    #                                                                             ep_b = eps,
    #                                                                             ep_a = eps)
    #     acc_rob_param_act_list.append(acc_rob_param_act) 

    #     acc_rob_input_param, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i =eps, 
    #                                                                             ep_w = eps,
    #                                                                             ep_b = eps,
    #                                                                             ep_a = 0)
    #     acc_rob_input_param_list.append(acc_rob_input_param) 

    #     acc_rob, _ = print_accuracy(net, trainloader, testloader, device, test=True, 
    #                                                                             ep_i = eps, 
    #                                                                             ep_w = eps,
    #                                                                             ep_b = eps,
    #                                                                             ep_a = eps)
    #     acc_rob_list.append(acc_rob) 


    # result = {'Epsilon': eps_list, 'Guaranteed accuracy with input perturbation': acc_rob_input_list, 
    #                             'Guaranteed accuracy with paramater perturbation': acc_rob_param_list,
    #                             'Guaranteed accuracy with activation perturbation': acc_rob_act_list, 
    #                             'Guaranteed accuracy with input and weight perturbation': acc_rob_input_param_list,
    #                             'Guaranteed accuracy with input and activation perturbation': acc_rob_input_act_list,
    #                             'Guaranteed accuracy with activation and weight perturbation': acc_rob_param_act_list,
    #                             'Guaranteed accuracy with activation, weight and activatsion perturbation': acc_rob_list}
    # path = f'results/MNIST/normal_6_layers_{n_hidden_nodes}_0_03.xlsx'
    # DictExcelSaver.save(result,path)
if __name__=="__main__":
    main()