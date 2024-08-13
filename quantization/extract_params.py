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

from interval_bound_propagation.network import *
from interval_bound_propagation.utils import progress_bar
from interval_bound_propagation.utils import generate_kappa_schedule_MNIST
from interval_bound_propagation.utils import generate_epsilon_schedule_MNIST
from interval_bound_propagation.compute_acc import *

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


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# print("Model State Dictionary:")
# for param_tensor in net.state_dict():
#     print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

folder_name = 'extracted_params/'
# Create directory if it doesn't exist
os.makedirs(folder_name, exist_ok=True)
# name = 'conv1.'
# conv_weight = net.state_dict()['module.conv1.weight'].cpu().numpy()
# bn_gamma = net.state_dict()['module.bn1.weight'].cpu().numpy()
# bn_beta = net.state_dict()['module.bn1.bias'].cpu().numpy()
# bn_rm = net.state_dict()['module.bn1.running_mean'].cpu().numpy()
# bn_rv = net.state_dict()['module.bn1.running_var'].cpu().numpy()

# factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
# np.save(folder_name+name+'weight.npy',conv_weight*factor[:,None,None,None])
# np.save(folder_name+name+'bias.npy',bn_beta-bn_rm*factor)
# print('done with '+name)

# for l in ['1','2','3','4']:
#     for s in ['0','1']:
#         for c in ['1','2']:
#             name = 'layer'+l+'.'+s+'.'
#             conv_weight = net.state_dict()['module.'+name+'conv'+c+'.weight'].cpu().numpy()
#             bn_gamma = net.state_dict()['module.'+name+'bn'+c+'.weight'].cpu().numpy()
#             bn_beta = net.state_dict()['module.'+name+'bn'+c+'.bias'].cpu().numpy()
#             bn_rm = net.state_dict()['module.'+name+'bn'+c+'.running_mean'].cpu().numpy()
#             bn_rv = net.state_dict()['module.'+name+'bn'+c+'.running_var'].cpu().numpy()
#             factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
#             np.save(folder_name+name+'conv'+c+'.weight.npy',conv_weight*factor[:,None,None,None])
#             np.save(folder_name+name+'conv'+c+'.bias.npy',bn_beta-bn_rm*factor)
#             print('done with '+name)
#         if (l!='1') and (s=='0'):
#             name = 'layer'+l+'.'+s+'.shortcut.'
#             conv_weight = net.state_dict()['module.'+name+'0.weight'].cpu().numpy()
#             bn_gamma = net.state_dict()['module.'+name+'1.weight'].cpu().numpy()
#             bn_beta = net.state_dict()['module.'+name+'1.bias'].cpu().numpy()
#             bn_rm = net.state_dict()['module.'+name+'1.running_mean'].cpu().numpy()
#             bn_rv = net.state_dict()['module.'+name+'1.running_var'].cpu().numpy()
#             factor = np.true_divide(bn_gamma,np.sqrt(bn_rv+1e-5))
#             np.save(folder_name+name+'weight.npy',conv_weight*factor[:,None,None,None])
#             np.save(folder_name+name+'bias.npy',bn_beta-bn_rm*factor)
#             print('done with '+name)
name = 'linear_layers'
for i in ['1', '3', '5']: 
    np.save(folder_name+name+i+'.weight.npy',net.state_dict()['module.layers.'+i+'.weight'].cpu().numpy())
    np.save(folder_name+name+i+'.bias.npy',net.state_dict()['module.layers.'+i+'.bias'].cpu().numpy())
print('done with '+name)

