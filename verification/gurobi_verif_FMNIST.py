import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import gurobipy as gp
from gurobipy import GRB
import torch.backends.cudnn as cudnn
import numpy
from utils import DictExcelSaver

import sys

new_path = "C:/Users/hueda/Documents/Model_robust_weight_perturbation"
sys.path.append(new_path) 
from interval_bound_propagation.network import *


# #print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#FMNIST
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])

# testset = torchvision.datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# net =  FMNIST_MLP(
#     non_negative = [False, False, False,False, False, False], 
#     norm = [False, False, False, False, False, False])

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
# net = net.to(device)


# # Load checkpoint.
# checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\FMNIST\running_eps_1_255.pth')
# net.load_state_dict(checkpoint['net'])

#MNIST
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

#Model
print('==> Building model..')
net =  MNIST_MLP(
    non_negative = [False, False, False], 
    norm = [False, False, False])

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

# Load checkpoint.
checkpoint = torch.load(r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\interval_bound_propagation\checkpoint\MNIST\running_eps_1_255.pth')
net.load_state_dict(checkpoint['net'])

#print(len(bounds))
# for i, (ub, lb) in enumerate(bounds):
#     print(f"Bound {i} upper_bound shape: {ub.shape}")
#     print(f"Bound {i} lower_bound shape: {lb.shape}")
# for i in [2,4,5]: 
#     upper_bound = bounds[i][0]  
#     lower_bound = bounds[i][1] 
#     #mse_loss+= torch.sqrt(nn.MSELoss()(upper_bound, lower_bound)+1e-6)
#     mse_loss+= nn.MSELoss()(upper_bound, lower_bound)

#load numpy files
model_dictionary = {}
folder_name = r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\quantization\extracted_params\MNIST\running_1_255/'

name = 'linear_layers'
for i in ['1', '2', '3']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    #print(model_dictionary[name+i+'weight'].size())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
    #print('done with '+name+i)

print('Check model dictionary')

import numpy
def feedforward(x,model_dictionary):
    linear_input = x.view(x.size(0),-1)
    for i in ['1', '2']: 
        name = 'linear_layers'+i
        linear_output = torch.matmul((model_dictionary[name+'weight']),linear_input.transpose(0,1))+model_dictionary[name+'bias'][:, None]
        linear_output = F.relu(linear_output)
        linear_input = linear_output.transpose(0, 1)
        
    #output layer 
    linear_output = torch.matmul(model_dictionary['linear_layers3'+'weight'],linear_input.transpose(0,1))+model_dictionary['linear_layers3'+'bias'][:, None]
    #y = F.softmax(linear_output, dim=0)
    return linear_output

def test_robustness(model_dictionary, net, testloader, epsilon_input=1/255, epsilon_weight=1/255, epsilon_bias=1/255, epsilon_activation=1/255):
    #for batch_idx, (inputs, targets) in enumerate(testloader):
    for batch_idx,  (inputs, labels)  in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # Flatten image to a vector
        labels = labels.item()

        #  Gurobi model
        model_gurobi = gp.Model("MLP_Robustness")

        # input variables
        x = model_gurobi.addVars(inputs.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        #print(inputs.shape[1])
        #  weight và bias variables (for layer 1-3)
        weight1 = model_gurobi.addVars(128, inputs.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight1")
        bias1 = model_gurobi.addVars(128, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias1")
        weight2 = model_gurobi.addVars(128, 128, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight2")
        bias2 = model_gurobi.addVars(128, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias2")
        weight3 = model_gurobi.addVars(10, 128, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight3")
        bias3 = model_gurobi.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias3")


        # w = []
        # b = []
        # linear_output = []
        # for i in ['1', '2', '3']: 
        #     name = 'linear_layers'+i
        #     m, n = model_dictionary[name+'weight'].size()
        #     print(m,n)
        #     w.append(model_gurobi.addVars(m,n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w"+i))
        #     b.append(model_gurobi.addVars(model_dictionary[name+'bias'].size(),lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b"+i))

        # input, weight and bias constraints
        for i in range(inputs.shape[1]):
            model_gurobi.addConstr(x[i] >= inputs[0][i].item() - epsilon_input, f"x_lower_{i}") #just 1 input each batch
            model_gurobi.addConstr(x[i] <= inputs[0][i].item() + epsilon_input, f"x_upper_{i}")
        # for i in range (len(w)):
        #     name = 'linear_layers'+str(i+1)
        #     for j in range(len(w[i])):  
        #         for k in range(len(w[i][j])): 
        #             model_gurobi.addConstr(w[i,j,k] >= model_dictionary[name+'weight'][j, k].item() - epsilon_weight, f"w_lower_{i}_{j}_{k}")
        #             model_gurobi.addConstr(w[i,j,k] <= model_dictionary[name+'weight'][j, k].item() + epsilon_weight, f"w_upper_{i}_{j}_{k}")

        # for i in range (len(b)):
        #     name = 'linear_layers'+i
        #     for j in range(len(b[i])):  
        #             model_gurobi.addConstr(b[i,j] >= model_dictionary[name+'bias'][j].item() - epsilon_bias, f"b_lower_{i}_{j}")
        #             model_gurobi.addConstr(b[i,j] <= model_dictionary[name+'bias'][j].item() + epsilon_bias, f"b_upper_{i}_{j}")

        # # define the linear output variables
        # for i in range(len(w)): 
        #     linear_output = gp.quicksum(w1[i] * x[i] for i in range(inputs.shape[1])) + bias1

        # # Hàm kích hoạt ReLU
        # a = model_gurobi.addVar(vtype=GRB.BINARY, name="a")  # Biến nhị phân cho ReLU
        # activation = model_gurobi.addVar(lb=0, ub=GRB.INFINITY, name="activation")
        
        # # Encode Relu
        # model_gurobi.addConstr(activation >= 0)
        # model_gurobi.addConstr(activation >= linear_output)
        # model_gurobi.addConstr(activation <= a * GRB.INFINITY)
        # model_gurobi.addConstr(activation <= linear_output + (1 - a) * GRB.INFINITY)
        
        # # activation constraints
        # model_gurobi.addConstr(activation >= linear_output - epsilon_activation, "activation_lower")
        # model_gurobi.addConstr(activation <= linear_output + epsilon_activation, "activation_upper")


        # #difine lower bound and upper bound to Encode ReLU constraints
        # bounds = net.module.bound_propagation(torch.cat([x_ub, x_lb], 0), epsilon_w=ep_w_schedule[batch_counter],
        #                                                 epsilon_b=ep_b_schedule[batch_counter],
        #                                                 epsilon_a=ep_a_schedule[batch_counter])

        # # Đặt hàm mục tiêu là tối ưu hóa activation tensor (tìm max/min)
        # model_gurobi.setObjective(activation, GRB.MAXIMIZE)
        # model_gurobi.optimize()
        # print(f"Max activation for sample {batch_idx}: {activation.X}")

        # model_gurobi.setObjective(activation, GRB.MINIMIZE)
        # model_gurobi.optimize()
        # print(f"Min activation for sample {batch_idx}: {activation.X}")

        # # Break sau một mẫu để kiểm tra
        # break

if __name__ == '__main__':
    test_robustness(model_dictionary,net, testloader, epsilon_input=1/255, epsilon_weight=1/255, epsilon_bias=1/255, epsilon_activation=1/255)