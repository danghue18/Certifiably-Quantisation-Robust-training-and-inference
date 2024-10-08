from __future__ import print_function

import torch

import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from network import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
indices = list(range(0, 100))
testloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=indices, num_workers=2)
#testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

classes = ('0','1','2','3','4','5','6','7','8','9')

n_hidden_nodes = 64
net =  MNIST_3layers(
    non_negative = [False, False, False], 
    norm = [False, False, False], 
    n_hidden_nodes=n_hidden_nodes )
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

# Load checkpoint.
checkpoint_path = f'C:/Users/hueda/Documents/Model_robust_weight_perturbation/interval_bound_propagation/checkpoint/MNIST/normal_3_layers_{n_hidden_nodes}.pth'
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])


# # Get the first 100 labels from the test set
# first_100_labels = [testset[i][1] for i in range(100)]

# # Count occurrences of each class (0-9)
# class_distribution = torch.bincount(torch.tensor(first_100_labels), minlength=10)

# # Print the class distribution
# for i, count in enumerate(class_distribution):
#     print(f"Class {i}: {count} images")
ep_i = 1/1023
ep_w = 1/1023
ep_b = 1/1023
ep_a = 1/1023
if __name__ == '__main__':
    fl_model_outputs = []
    worst_case_diff = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.item()

            outputs = net(torch.cat([inputs,inputs], 0))
            value_fl_batch = outputs[:outputs.shape[0]//2]
            value_fl_batch = F.softmax(value_fl_batch, dim=1)
            orig_output = value_fl_batch[0,targets].item() # output of true class
            # print((np.array(fl_model_outputs)).shape)
            #print(fl_model_outputs)

            x_ub = inputs + ep_i
            x_lb = inputs - ep_i
            
            outputs = net(torch.cat([x_ub,x_lb], 0), ep_w, ep_b, ep_a)
            #outputs = F.softmax(outputs, dim=1)
            #print(outputs)
            #print(outputs.shape)
            outputs_ub = outputs[:outputs.shape[0]//2]
            #print(outputs_ub.shape)
            outputs_lb = outputs[outputs.shape[0]//2:]

            best_case = outputs_lb.clone()
            best_case[0,targets] = outputs_ub[0,targets]
            best = (F.softmax(best_case, dim=1))[0,targets].item()

            worst_case = outputs_ub.clone()
            worst_case[0,targets] = outputs_lb[0,targets]
            #print(F.softmax(worst_case, dim=1))
            worst =(F.softmax(worst_case, dim=1))[0,targets].item()
            

            diff =  max(best - orig_output, orig_output - worst)
            #diff =  max(outputs_ub[0,targets] - orig_output, orig_output - outputs_lb[0,targets])
            if diff > worst_case_diff: 
                worst_case_diff = diff
    print(worst_case_diff)