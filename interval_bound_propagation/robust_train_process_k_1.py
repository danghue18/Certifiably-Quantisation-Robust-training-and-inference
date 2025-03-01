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
from utils import generate_epsilon_fixed_MNIST
from compute_acc import *
from utils import DictExcelSaver

from multiprocessing import freeze_support


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=0.5, type=float, help='kappa')
parser.add_argument('--ep_i', default=1/128, type=float, help='epsilon_input')
parser.add_argument('--ep_w', default=1/64, type=float, help='epsilon_weight')
parser.add_argument('--ep_b', default=1/64, type=float, help='epsilon_bias')
parser.add_argument('--ep_a', default=1/64, type=float, help='epsilon_activation')
parser.add_argument('--ep_scheme', action='store_false', help='disable epsilon strategy (default: enabled)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print('Using device: ', device)
best_acc = 0  # best test accuracy
nor_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_nor_list = []
acc_rob_list = []
k_list = []
eps_list = []

train_fit_loss_list = []
train_robust_loss_list = []
train_loss_list = []

val_fit_loss_list = []
val_robust_loss_list = []
val_loss_list = []

# Data
#print('==> Preparing data..')
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

# Model
#print('==> Building model..')
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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#classes = ('0','1','2','3','4','5','6','7','8','9')

#Model
print('==> Building model..')
n_hidden_nodes = 256
net =  MNIST_4layers(
    non_negative = [False, False, False,False], 
    norm = [False, False, False, False], 
    n_hidden_nodes=n_hidden_nodes )
    # non_negative = [True, True, True], 
    # norm = [True, True, True])
    # non_negative = [True, True, True], 
    # norm = [False, False, False])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

ep_i = args.ep_i
ep_w = args.ep_w
ep_b = args.ep_b
ep_a = args.ep_a
k = args.k
kappa_schedule = generate_kappa_schedule_MNIST(k)

if args.ep_scheme:
    ep_i_schedule = generate_epsilon_fixed_MNIST(ep_i)
    ep_w_schedule = generate_epsilon_fixed_MNIST(ep_w)
    ep_b_schedule = generate_epsilon_fixed_MNIST(ep_b)
    ep_a_schedule = generate_epsilon_fixed_MNIST(ep_a)
else:   
    ep_i_schedule = generate_epsilon_schedule_MNIST(ep_i)
    ep_w_schedule = generate_epsilon_schedule_MNIST(ep_w)
    ep_b_schedule = generate_epsilon_schedule_MNIST(ep_b)
    ep_a_schedule = generate_epsilon_schedule_MNIST(ep_a)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/MNIST/robust_4_layers_{n_hidden_nodes}_{k}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc_rob']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch, batch_counter):
    print('\nEpoch: %d' % epoch)
    print('\nBatch_counter: %d' % batch_counter)
    net.train()
    train_loss = 0 
    train_robust_loss = 0
    train_fit_loss = 0
    # correct = 0
    # total = 0
    global best_acc
    global nor_acc
    global acc_nor_list 
    global acc_rob_list 
    global k_list
    global eps_list 

    global train_fit_loss_list
    global train_robust_loss_list
    global train_loss_list
    
    global val_fit_loss_list
    global val_robust_loss_list
    global val_loss_list
                                                                                                                                                                                                                                                                                                                                   
    lr =args.lr
    if epoch>50:
        lr/=10
    if epoch>75:    
        lr/=10
    if epoch>100: 
        lr/=10
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        loss = 0 # to update params
        #print(ep_i_schedule[batch_counter])

        optimizer.zero_grad()
        

        outputs_nor = net(torch.cat([inputs, inputs], 0))
        outputs_nor  = outputs_nor[:outputs_nor.shape[0]//2]
        loss += kappa_schedule[batch_counter] * criterion(outputs_nor, targets)
        fit_loss = criterion(outputs_nor, targets)

        if batch_counter >= 3000: 
            x_ub = inputs + ep_i_schedule[batch_counter]
            x_lb = inputs - ep_i_schedule[batch_counter]
            outputs = net.forward(torch.cat([x_ub, x_lb], 0), epsilon_w=ep_w_schedule[batch_counter],
                                                              epsilon_b=ep_b_schedule[batch_counter],
                                                              epsilon_a=ep_a_schedule[batch_counter])                                 

            z_ub = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            lb_mask = torch.eye(10).to(device)[targets] # one hot encoding of true label
            ub_mask = 1 - lb_mask 
            outputs = z_lb * lb_mask + z_ub * ub_mask # z_lb is in true label position, z_ub in other positions 
            loss += (1-kappa_schedule[batch_counter]) * criterion(outputs, targets)
            
        # compute train loss
        x_ub = inputs + ep_i_schedule[batch_counter]
        x_lb = inputs - ep_i_schedule[batch_counter]
        outputs = net.forward(torch.cat([x_ub, x_lb], 0),   epsilon_w=ep_w,
                                                            epsilon_b=ep_b,
                                                            epsilon_a=ep_a) 
        z_ub = outputs[:outputs.shape[0]//2]
        z_lb = outputs[outputs.shape[0]//2:]
        lb_mask = torch.eye(10).to(device)[targets] # one hot encoding of true label
        ub_mask = 1 - lb_mask 
        outputs = z_lb * lb_mask + z_ub * ub_mask # z_lb is in true label position, z_ub in other positions 
        robust_loss =  criterion(outputs, targets)

        train_robust_loss +=robust_loss.item()
        train_fit_loss +=fit_loss.item()
        train_loss += (kappa_schedule[batch_counter] * fit_loss + (1-kappa_schedule[batch_counter]) * robust_loss).item()
        #print(kappa_schedule[batch_counter])
        loss.backward()
        optimizer.step()
        batch_counter+=1


    
    # print('1***',train_fit_loss/600)
    # print('2***',train_robust_loss/600)
    # print('3***',train_loss/600)

    train_fit_loss_list.append(train_fit_loss/600)
    train_robust_loss_list.append(train_robust_loss/600)
    train_loss_list.append(train_loss/600)
    #print('Total loss in training: %.3f' % (train_loss / 600))


    net.eval()
    #print('Loss computed print_acc function:',l+loss_train)

    _,l = print_accuracy(net, trainloader, testloader, device, test=False, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
    #print('4***',l)
    _, l = print_accuracy(net, trainloader, testloader, device, test=False, ep_i = args.ep_i, 
                                                                            ep_w=args.ep_w,
                                                                            ep_b=args.ep_b,
                                                                            ep_a=args.ep_a)
    #print('5***',l)

    acc_nor,val_fit_loss = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0)
    acc_nor_list.append(acc_nor)
    val_fit_loss_list.append(val_fit_loss)

    acc_rob, val_robust_loss = print_accuracy(net, trainloader, testloader, device, test=True, ep_i = args.ep_i, 
                                                                            ep_w=args.ep_w,
                                                                            ep_b=args.ep_b,
                                                                            ep_a=args.ep_a)
    acc_rob_list.append(acc_rob)
    val_robust_loss_list.append(val_robust_loss)
    val_loss = val_robust_loss*(1-k) + val_fit_loss*k
    val_loss_list.append(val_loss)

    if acc_rob > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_nor': acc_nor,
            'acc_rob': acc_rob,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/MNIST/epchange/'):
            os.mkdir('checkpoint/MNIST/epchange')
        torch.save(state, f'./checkpoint/MNIST/epchange/robust_4_layers_{n_hidden_nodes}_{k}_{ep_i}_0.pth')
        best_acc = acc_rob
        print("best_acc: ", best_acc)
        nor_acc = acc_nor
        print("nor_acc: ", nor_acc)

    if not os.path.isdir('results/training_phase/MNIST/changek/'):
        os.mkdir('results/training_phase/MNIST/changek/')
    result={'train fit loss': train_fit_loss_list,'train robust loss': train_robust_loss_list, 'train loss': train_loss_list,
            'val fit loss': val_fit_loss_list, 'val robust loss': val_robust_loss_list, 'val loss': val_loss_list}
    path = f'results/training_phase/MNIST/changeep/robust_4_layers_{n_hidden_nodes}_{k}_{ep_i}_0.xlsx'
    DictExcelSaver.save(result,path)
    epoch+= 1
    #1 = no 'ep_scheme' = fixed
    #0 =  'ep_scheme' = running 



if __name__=="__main__":
    freeze_support()  # This is necessary for Windows when using multiprocessing
    batch_counter = 0
    for epoch in range(start_epoch, start_epoch+125):
        train(epoch, batch_counter)
        batch_counter+=600
  
    print ("best_acc: ", best_acc)
    print("nor_acc: ", nor_acc)
    

    # result={'train fit loss': train_fit_loss_list,'train robust loss': train_robust_loss_list, 'train loss': train_loss_list,
    #         'val fit loss': val_fit_loss_list, 'val robust loss': val_robust_loss_list, 'val loss': val_loss_list}
    # path = f'results/training_phase/MNIST/changek/robust_6_layers_{n_hidden_nodes}_{k}_{ep_i}.xlsx'
    # DictExcelSaver.save(result,path)



