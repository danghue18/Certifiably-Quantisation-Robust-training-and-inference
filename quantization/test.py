
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
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from interval_bound_propagation.network import *
from interval_bound_propagation.utils import progress_bar
from interval_bound_propagation.utils import generate_kappa_schedule_MNIST
from interval_bound_propagation.utils import generate_epsilon_schedule_MNIST
from interval_bound_propagation.compute_acc import *

from multiprocessing import freeze_support


#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

# Function to dequantize weights
import torch
import torch.nn.functional as F

# Hàm thực hiện dequantize đúng cách
def dequantize_weights_lsq(state_dict, num_bits=9):
    """Dequantize weights using LSQ approach."""
    dequantized_state_dict = {}
    qmin = -(2**(num_bits - 1))
    qmax = 2**(num_bits - 1) - 1

    for key, value in state_dict.items():
        if 'weight' in key:
            # Lấy tên lớp (ví dụ: "fc6" từ "fc6.weight")
            layer_prefix = key.split('.')[0]
            
            # Kiểm tra xem alpha tồn tại hay không
            alpha_key = f"{layer_prefix}.alpha"
            if alpha_key in state_dict:
                alpha = state_dict[alpha_key]
                # Thực hiện dequantize
                weight = value / alpha  # Lấy lại trọng số trước lượng tử hóa
                weight = torch.clamp(weight, qmin, qmax)  # Clamp theo range lượng tử hóa
                dequantized_state_dict[key] = weight * alpha  # Tái tạo trọng số
            else:
                dequantized_state_dict[key] = value  # Copy nguyên bản nếu không có alpha
        else:
            dequantized_state_dict[key] = value  # Copy bias hoặc alpha nguyên bản
    
    return dequantized_state_dict

# Hàm forward pass chính xác
def feedforward_lsq(x, model_dictionary, num_bits=9):
    """Perform inference with dequantized weights."""
    qmin = -(2**(num_bits - 1))
    qmax = 2**(num_bits - 1) - 1
    linear_input = x.view(x.size(0), -1)  # Flatten input
    
    for i in range(1, 6):  # Layers 1 to 5 with ReLU activations
        weight = model_dictionary[f'fc{i}.weight']
        bias = model_dictionary[f'fc{i}.bias']
        alpha_key = f'fc{i}.alpha'
        
        # Lượng tử hóa trọng số và thực hiện forward pass
        if alpha_key in model_dictionary:
            alpha = model_dictionary[alpha_key]
            weight_quant = torch.clamp((weight / alpha).round(), qmin, qmax) * alpha
        else:
            weight_quant = weight
        
        linear_output = F.linear(linear_input, weight_quant, bias)
        linear_input = F.relu(linear_output)
    
    # Output layer (không activation)
    weight = model_dictionary['fc6.weight']
    bias = model_dictionary['fc6.bias']
    alpha_key = 'fc6.alpha'
    
    if alpha_key in model_dictionary:
        alpha = model_dictionary[alpha_key]
        weight_quant = torch.clamp((weight / alpha).round(), qmin, qmax) * alpha
    else:
        weight_quant = weight
    
    linear_output = F.linear(linear_input, weight_quant, bias)
    return linear_output

# Hàm test
def test_lsq(model_dictionary, testloader, num_bits=9, device='cuda'):
    """Evaluate accuracy of the dequantized model."""
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = feedforward_lsq(inputs, model_dictionary, num_bits)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc




if __name__ == '__main__':
    freeze_support()
# Tải checkpoint QAT
    checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\RobustQuantization\nn_quantization_pytorch\nn-quantization-pytorch\quantization\qat\mxt-sim\ckpt\mlp_cifar10\2024-12-04_16-56-46\model_best.pth.tar')
    qat_state_dict = checkpoint['state_dict']

    # Dequantize đúng cách
    dequantized_state_dict = dequantize_weights_lsq(qat_state_dict, num_bits=2)

    # Chuyển thành dictionary để inference
    model_dictionary = {k: v.cuda() for k, v in dequantized_state_dict.items()}

    # Test dequantized model
    test_accuracy = test_lsq(model_dictionary, testloader, num_bits=2, device='cuda')
    print(f"Test Accuracy: {test_accuracy:.2f}%")
