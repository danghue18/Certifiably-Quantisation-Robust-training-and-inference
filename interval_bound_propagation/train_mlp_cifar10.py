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

from network import MLP_cifar10
from utils import progress_bar
from compute_acc import *

from multiprocessing import freeze_support


#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
net =  MLP_cifar10()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    #print('\nBatch_counter: %d' % batch_counter)
    net.train()
    train_loss = 0
    # correct = 0
    # total = 0
    global best_acc
    global rob_acc
                                                                                                                                                                                                                                                                                                                                               
    lr = 0.1
    if epoch>50:
        lr/=10
    if epoch>75:
        lr/=10
    if epoch>100: 
        lr/=10
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_nor = net(inputs)
        loss =  criterion(outputs_nor, targets)


        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    print('Loss in training: %.3f' % (train_loss / 600))
    net.eval()
    accuracy(net, trainloader, testloader, device, test=False)
    acc_nor = accuracy(net, trainloader, testloader, device, test=True)


    if acc_nor > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_nor': acc_nor,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/CIFAR10/'):
            os.mkdir('checkpoint/CIFAR10')
        torch.save(state, './checkpoint/CIFAR10/pretrained_mlp.pth')
        best_acc = acc_nor
        print("best_acc: ", best_acc)
    
    epoch+= 1

def accuracy(net, trainloader, testloader, device, test=True):
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

            outputs = net(images)
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

    print('Accuracy of the network on the', total, loadertype, 'images: ',correct)
    return correct

if __name__=="__main__":
    freeze_support()  # This is necessary for Windows when using multiprocessing
    for epoch in range(start_epoch, start_epoch+125):
        train(epoch)
  
    print ("best_acc: ", best_acc)
