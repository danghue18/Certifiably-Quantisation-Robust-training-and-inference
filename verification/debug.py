import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import gurobipy as gp
from gurobipy import GRB
import torch.backends.cudnn as cudnn
import numpy

import sys

new_path = "C:/Users/hueda/Documents/Model_robust_weight_perturbation"
sys.path.append(new_path) 
from interval_bound_propagation.network import *


# #print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

#load numpy files
model_dictionary = {}
folder_name = r'C:\Users\hueda\Documents\Model_robust_weight_perturbation\quantization\extracted_params\MNIST\running_1_255/'

# load trained weights and bias in model_dictionary
name = 'linear_layers'
for i in ['1', '2', '3']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(folder_name+name+i+'.weight.npy')).cuda()
    # print(model_dictionary[name+i+'weight'].size())
    # print(model_dictionary[name+i+'weight'][0][1].item())
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(folder_name+name+i+'.bias.npy')).cuda()
    #print('done with '+name+i)


def test_robustness(model_dictionary, net, testloader, epsilon_input=1/255, epsilon_weight=1/255, epsilon_bias=1/255, epsilon_activation=1/255):
    #for batch_idx, (inputs, targets) in enumerate(testloader):
    robust_count = 0
    total = 0
    for batch_idx,  (inputs, labels)  in enumerate(testloader, 0):
        #define lower bound and upper bound of each linear layer output
        x_ub = inputs.to(device) + epsilon_input
        x_lb = inputs.to(device) - epsilon_input
        bounds = net.module.linear_bound(torch.cat([x_ub, x_lb], 0), epsilon_weight, epsilon_bias, epsilon_activation)
        
        inputs = inputs.view(inputs.size(0), -1)  # Flatten image to a vector
        labels = labels.item()
        print(labels)

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

        # input constraints
        for i in range(inputs.shape[1]):
            model_gurobi.addConstr(x[i] >= inputs[0][i].item() - epsilon_input, f"x_lower_{i}") #just 1 input each batch
            model_gurobi.addConstr(x[i] <= inputs[0][i].item() + epsilon_input, f"x_upper_{i}")
        # weight constraints
        name = 'linear_layers'
        for i in range(128): 
            for j in range(inputs.shape[1]): 
                model_gurobi.addConstr(weight1[i,j] >= model_dictionary[name+'1weight'][i][j].item() - epsilon_weight, f"w1_lower_{i}_{j}")
                model_gurobi.addConstr(weight1[i,j] <= model_dictionary[name+'1weight'][i][j].item() + epsilon_weight, f"w1_upper_{i}_{j}")
        for i in range(128): 
            for j in range(128): 
                model_gurobi.addConstr(weight2[i,j] >= model_dictionary[name+'2weight'][i][j].item() - epsilon_weight, f"w2_lower_{i}_{j}")
                model_gurobi.addConstr(weight2[i,j] <= model_dictionary[name+'2weight'][i][j].item() + epsilon_weight, f"w2_upper_{i}_{j}")
        for i in range(10): 
            for j in range(128): 
                model_gurobi.addConstr(weight3[i,j] >= model_dictionary[name+'3weight'][i][j].item() - epsilon_weight, f"w3_lower_{i}_{j}")
                model_gurobi.addConstr(weight3[i,j] <= model_dictionary[name+'3weight'][i][j].item() + epsilon_weight, f"w3_upper_{i}_{j}")
        #bias constraints 
        for i in range(128): 
            model_gurobi.addConstr(bias1[i] >= model_dictionary[name+'1bias'][i].item() - epsilon_bias, f"b1_lower_{i}")
            model_gurobi.addConstr(bias1[i] <= model_dictionary[name+'1bias'][i].item() + epsilon_bias, f"b1_upper_{i}")
        for i in range(128): 
            model_gurobi.addConstr(bias2[i] >= model_dictionary[name+'2bias'][i].item() - epsilon_bias, f"b2_lower_{i}")
            model_gurobi.addConstr(bias2[i] <= model_dictionary[name+'2bias'][i].item() + epsilon_bias, f"b2_upper_{i}")
        for i in range(10): 
            model_gurobi.addConstr(bias3[i] >= model_dictionary[name+'3bias'][i].item() - epsilon_bias, f"b3_lower_{i}")
            model_gurobi.addConstr(bias3[i] <= model_dictionary[name+'3bias'][i].item() + epsilon_bias, f"b3_upper_{i}")
        print("----------------done constraints for input, weights, bias")
        

        #print(len(bounds))
        # for i, (ub, lb) in enumerate(bounds):
        #     print(f"Bound {i} upper_bound shape: {ub.shape}")
        #     print(f"Bound {i} lower_bound shape: {lb.shape}")
            # upper_bound = bounds[i][0]  
            # lower_bound = bounds[i][1] 
            # print(bounds[i][0][0])

        # first layer (128 neurons)
        linear_output1 = model_gurobi.addVars(128, name="linearoutput1") # define the linear output variables
        activation1 = model_gurobi.addVars(128, name="activation1") #relu output - input of next layer
        #a1 = model_gurobi.addVar(vtype=GRB.BINARY, name="a1")  # binary variable for relu
        upper_bound1 = bounds[0][0] # 128 upper bounds
        lower_bound1 = bounds[0][1] # 128 lower bounds 
        for i in range(128):
            model_gurobi.addConstr(linear_output1[i] == gp.quicksum([weight1[i, j] * x[j] for j in range(inputs.shape[1])]) + bias1[i])

            # model_gurobi.addConstr(activation1[i] <= linear_output1[i] - (1 - a1) * lower_bound1[0,i].item())
            # model_gurobi.addConstr(activation1[i] >= linear_output1[i])
            # model_gurobi.addConstr(activation1[i] <= a1 * upper_bound1[0, i].item())
            # model_gurobi.addConstr(activation1[i] >= 0)  
            # print(" ------lower bound: ", lower_bound1[0,i].item())
            # print(" ------upper bound: ", upper_bound1[0,i].item())
            
        
            model_gurobi.addConstr(activation1[i] >= linear_output1[i] - epsilon_activation, "activation1_lower")
            model_gurobi.addConstr(activation1[i] <= linear_output1[i] + epsilon_activation, "activation1_upper")

        # layer 2 (128 neurons)
        linear_output2 = model_gurobi.addVars(128,lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput2") # define the linear output variables
        activation2 = model_gurobi.addVars(128, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="activation2") #relu output - input of next layer
        #a2 = model_gurobi.addVar(vtype=GRB.BINARY, name="a2")  # binary variable for relu
        upper_bound2 = bounds[1][0] # 128 upper bounds
        lower_bound2 = bounds[1][1] # 128 lower bounds 
        for i in range(128):
            model_gurobi.addConstr(linear_output2[i] == gp.quicksum([weight2[i, j] * activation1[j] for j in range(128)]) + bias2[i])
            
            # model_gurobi.addConstr(activation2[i] <= linear_output2[i] - (1 - a2) * lower_bound2[0,i].item())
            # model_gurobi.addConstr(activation2[i] >= linear_output2[i])  
            # model_gurobi.addConstr(activation2[i] <= a2 * upper_bound2[0, i].item())
            # model_gurobi.addConstr(activation2[i] >= 0)  
        
            model_gurobi.addConstr(activation2[i] >= linear_output2[i] - epsilon_activation, "activation2_lower")
            model_gurobi.addConstr(activation2[i] <= linear_output2[i] + epsilon_activation, "activation2_upper")

        # output layer (10 neurons)
        output = model_gurobi.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="output")
        upper_bound3 = bounds[2][0] # 128 upper bounds
        lower_bound3 = bounds[2][1] # 128 lower bounds 
        for i in range(10):
            model_gurobi.addConstr(output[i] == gp.quicksum(weight3[i, j] * activation2[j] for j in range(128)) + bias3[i])
            model_gurobi.addConstr(output[i] >= lower_bound3[0,i].item())
            model_gurobi.addConstr(output[i] <= upper_bound3[0,i].item())

        # Objective function:  min_j≠t(P(f)(x)_t − P(f)(x)_j)
        max_other_class = model_gurobi.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="max_other_class")
        model_gurobi.addConstr(max_other_class == gp.max_([output[j] for j in range(10) if j != labels]))

        m = model_gurobi.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="m")
        model_gurobi.addConstr(m == output[labels] - max_other_class)

        # Optimize m
        model_gurobi.setObjective(m, GRB.MINIMIZE)
        model_gurobi.optimize()

        # Verify
        if model_gurobi.status == GRB.OPTIMAL:
            if m.X > 0:
                print(f"--------------Robust sample with m = {m.X}")
            else:
                print(f"--------------Non-robust sample with m = {m.X}")
        else:
            print("--------------Optimization was not successful.")

        #if model_gurobi.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model_gurobi.computeIIS()
        model_gurobi.write("model.ilp")

        # elif model_gurobi.status == GRB.UNBOUNDED:
        #     print("----------Model is unbounded")
        # elif model_gurobi.status == GRB.OPTIMAL:
        #     print("-------Model is optimal with objective value", model_gurobi.objVal)

        total += 1
        print('---------------done 1 sample')
        


    accuracy = 100 * robust_count / total
    print(f"Verified robust accuracy: {robust_count}/{total} ({accuracy}%)")

if __name__ == '__main__':
    test_robustness(model_dictionary,net, testloader, epsilon_input=1/255, epsilon_weight=1/255, epsilon_bias=1/255, epsilon_activation=1/255)