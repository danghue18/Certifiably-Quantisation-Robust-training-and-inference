from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import os
import argparse

from network import *
from utils import progress_bar
from utils import generate_kappa_schedule_SVHN
from utils import generate_epsilon_schedule_SVHN
from compute_acc import *

from multiprocessing import freeze_support


parser = argparse.ArgumentParser(description='PyTorch SVHN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ep_i', default=1/255, type=float, help='epsilon_input')
parser.add_argument('--ep_w', default=1/255, type=float, help='epsilon_weight')
parser.add_argument('--ep_b', default=1/255, type=float, help='epsilon_bias')
parser.add_argument('--ep_a', default=1/255, type=float, help='epsilon_activation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
net =  SVHN_MLP(
    non_negative = [False, False, False, False, False, False], 
    norm = [False, False, False,False, False, False])
    # non_negative = [True, True, True], 
    # norm = [True, True, True])
    # non_negative = [True, True, True], 
    # norm = [False, False, False])
net = net.to(device)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/normal'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\normal\ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc_nor']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

ep_i = args.ep_i
ep_w = args.ep_w
ep_b = args.ep_b
ep_a = args.ep_a
kappa_schedule = generate_kappa_schedule_SVHN()
ep_i_schedule = generate_epsilon_schedule_SVHN(ep_i)
ep_w_schedule = generate_epsilon_schedule_SVHN(ep_w)
ep_b_schedule = generate_epsilon_schedule_SVHN(ep_b)
ep_a_schedule = generate_epsilon_schedule_SVHN(ep_a)

# Training
def train(epoch,batch_counter):
    print('\nEpoch: %d' % epoch)
    #print('\nBatch_counter: %d' % batch_counter)
    net.train()
    train_loss = 0
    # correct = 0
    # total = 0
    global best_acc
    global rob_acc
                                                                                                                                                                                                                                                                                                                                               
    lr =args.lr
    if epoch>50:
        lr/=10
    if epoch>75:
        lr/=10
    if epoch>100: 
        lr/=10
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = StepLR(optimizer, step_size=, gamma=0.9)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        

        outputs_nor = net(torch.cat([inputs, inputs], 0))
        outputs_nor  = outputs_nor[:outputs_nor.shape[0]//2]
        loss =  criterion(outputs_nor, targets)


        loss.backward()
        optimizer.step()
        batch_counter+=1

        train_loss += loss.item()

    print('Loss in training: %.3f' % (train_loss / 600))
    net.eval()
    print_accuracy(net, trainloader, testloader, device, test=False, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
    acc_nor,_ = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
    acc_rob,_ = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 1/255, 
                                                                                  ep_w=1/255,
                                                                                  ep_b=1/255,
                                                                                  ep_a=1/255)

    if acc_nor > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_nor': acc_nor,
            'acc_rob': acc_rob,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/SVHN/'):
            os.mkdir('checkpoint/SVHN')
        torch.save(state, './checkpoint/SVHN/normal.pth')
        best_acc = acc_nor
        rob_acc = acc_rob
        print("best_acc: ", best_acc)
        print('robust_acc: ',rob_acc)
    
    epoch+= 1

if __name__=="__main__":
    freeze_support()  # This is necessary for Windows when using multiprocessing
    batch_counter = 0
    for epoch in range(start_epoch, start_epoch+125):
        train(epoch, batch_counter)
        batch_counter+=600
  
    print ("best_acc: ", best_acc)
    print('robust_acc: ', rob_acc)
