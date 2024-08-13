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
import matplotlib.pyplot as plt

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
nor_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_nor_list = []
acc_rob_list = []
k_list = []
eps_list = []
loss_train_list = []
loss_val_list = []

# Data
#print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
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


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc_rob']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

ep_i = args.ep_i
ep_w = args.ep_w
ep_b = args.ep_b
ep_a = args.ep_a
kappa_schedule = generate_kappa_schedule_MNIST()
ep_i_schedule = generate_epsilon_schedule_MNIST(ep_i)
ep_w_schedule = generate_epsilon_schedule_MNIST(ep_w)
ep_b_schedule = generate_epsilon_schedule_MNIST(ep_b)
ep_a_schedule = generate_epsilon_schedule_MNIST(ep_a)




# Training
def train(epoch, batch_counter):
    print('\nEpoch: %d' % epoch)
    print('\nBatch_counter: %d' % batch_counter)
    net.train()
    train_loss = 0
    # correct = 0
    # total = 0
    global best_acc
    global nor_acc
    global acc_nor_list 
    global acc_rob_list 
    global k_list
    global eps_list 
    global loss_train_list
    global loss_val_list

                                                                                                                                                                                                                                                                                                                                               
    lr =args.lr
    if epoch>50:
        lr/=10
    if epoch>75:
        lr/=10
    if epoch>100: 
        lr/=2
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        loss = 0
        optimizer.zero_grad()
        

        outputs_nor = net(torch.cat([inputs, inputs], 0))
        outputs_nor  = outputs_nor[:outputs_nor.shape[0]//2]
        loss += kappa_schedule[batch_counter] * criterion(outputs_nor, targets)

        if batch_counter >= 2400: 
            x_ub = inputs + ep_i_schedule[batch_counter]
            x_lb = inputs - ep_i_schedule[batch_counter]
            outputs = net.forward(torch.cat([x_ub, x_lb], 0), epsilon_w=ep_w_schedule[batch_counter],
                                                              epsilon_b=ep_b_schedule[batch_counter],
                                                              epsilon_a=ep_a_schedule[batch_counter])                     

            z_ub = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            lb_mask = torch.eye(10).cuda()[targets] # one hot encoding of true label
            ub_mask = 1 - lb_mask 
            outputs = z_lb * lb_mask + z_ub * ub_mask # z_lb is in true label position, z_ub in other positions 
            loss += (1-kappa_schedule[batch_counter]) * criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_counter+=1

        train_loss += loss.item()

    #print('Loss in training: %.3f' % (train_loss / 600))
    net.eval()
    if batch_counter < 2400: 
        _, loss_train = print_accuracy(net, trainloader, testloader, device, test=False, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        acc_nor, loss_val = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        acc_rob = acc_nor
    if batch_counter >= 2400 and batch_counter < 14400: 
        print_accuracy(net, trainloader, testloader, device, test=False, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        _, loss_train = print_accuracy(net, trainloader, testloader, device, test=False, ep_i = ep_i_schedule[batch_counter], 
                                                                        ep_w=ep_w_schedule[batch_counter],
                                                                        ep_b=ep_b_schedule[batch_counter],
                                                                        ep_a=ep_a_schedule[batch_counter])
        acc_nor,_ = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        acc_rob, loss_val = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = ep_i_schedule[batch_counter], 
                                                                        ep_w=ep_w_schedule[batch_counter],
                                                                        ep_b=ep_b_schedule[batch_counter],
                                                                        ep_a=ep_a_schedule[batch_counter])
    if batch_counter >= 14400:
        #     print_accuracy(net, trainloader, testloader, device, test=True, eps = 2/255)
        print_accuracy(net, trainloader, testloader, device, test=False, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        _, loss_train = print_accuracy(net, trainloader, testloader, device, test=False, ep_i = ep_i_schedule[batch_counter], 
                                                                         ep_w=ep_w_schedule[batch_counter],
                                                                         ep_b=ep_b_schedule[batch_counter],
                                                                         ep_a=ep_a_schedule[batch_counter])
        acc_nor,_ = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
        acc_rob, loss_val = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = ep_i_schedule[batch_counter], 
                                                                                  ep_w=ep_w_schedule[batch_counter],
                                                                                  ep_b=ep_b_schedule[batch_counter],
                                                                                  ep_a=ep_a_schedule[batch_counter])


        if acc_rob > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc_nor': acc_nor,
                'acc_rob': acc_rob,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint/test_k'):
                os.mkdir('checkpoint/test_k')
            torch.save(state, './checkpoint/test_k/ckpt.pth')
            best_acc = acc_rob
            print("best_acc: ", best_acc)
            nor_acc = acc_nor
            print("nor_acc: ", nor_acc)
    acc_nor_list.append(acc_nor)
    acc_rob_list.append(acc_rob)
    loss_train_list.append(loss_train)
    loss_val_list.append(loss_val)

    epoch+= 1
    #print('batch_counter after 1 epoch: ', batch_counter)
    k_list.append(kappa_schedule[batch_counter])
    eps_list.append(ep_i_schedule[batch_counter])

if __name__=="__main__":
    freeze_support()  # This is necessary for Windows when using multiprocessing
    batch_counter = 0
    for epoch in range(start_epoch, start_epoch+198):
        train(epoch, batch_counter)
        batch_counter+=600
  
    print ("best_acc: ", best_acc)
    print("nor_acc: ", nor_acc)
    


    result = {'acc_rob': acc_rob_list, 'acc_nor': acc_nor_list, 'loss_train': loss_train_list, 'loss_val': loss_val_list}
    path = 'results/training_phase/epsilon_strategy/running_eps_2_255.xlsx'
    DictExcelSaver.save(result,path)

    
    # epoch_list = [x for x in range(start_epoch, start_epoch + 198)]
    # plt.plot(epoch_list, acc_rob_list, label='robust_acc', marker='o')
    # plt.plot(epoch_list, acc_nor_list, label='normal_acc', marker='s')
    # # Add title and labels
    # plt.title('Accuracy in training process with k = 0.5')
    # plt.xlabel('epoch')
    # plt.ylabel('Accuracy')

    # # Show legend
    # plt.legend()

    # # Show grid
    # plt.grid(True)

    # # Display the graph
    # plt.show()


