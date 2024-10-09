import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import gurobipy as gp
from gurobipy import GRB
import torch.backends.cudnn as cudnn
import time

from dotenv import load_dotenv
import sys
import os

load_dotenv()


root = os.getenv('ROOT')
checkpoint_path = os.path.join(root, os.getenv("CHECKPOINT_PATH16"))
weight_folder = os.path.join(root,os.getenv('WEIGHT_FOLDER16'))

sys.path.append(root)
from interval_bound_propagation.network import *
from interval_bound_propagation.utils import DictExcelSaver



device = 'cuda' if torch.cuda.is_available() else 'cpu'
#MNIST
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(root='\datasets', train=False, download=True, transform=transform_test)

indices = list(range(0, 50))
testloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=indices, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# load net to compute original prediction and bounds of each linear output
n_hidden_nodes = 512
net =  MNIST_6layers(
    non_negative = [False, False, False, False, False, False], 
    norm = [False, False, False,False, False, False], 
    n_hidden_nodes=n_hidden_nodes )
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
else: 
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['net']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key.replace('module.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)


#load weight files
model_dictionary = {}

# load trained weights and biases in model_dictionary
name = 'linear_layers'
for i in ['1', '2', '3', '4', '5', '6']: 
    model_dictionary[name+i+'weight'] = torch.from_numpy(np.load(weight_folder+name+i+'.weight.npy')).to(device)
    model_dictionary[name+i+'bias']= torch.from_numpy(np.load(weight_folder+name+i+'.bias.npy')).to(device)
    #print('done with '+name+i)

# def optimize_with_timeout(model, timeout):
#     """
#     Run optimize model in timeout
#     If the optimiser can't answer in timeout seconds, stop the process and marked the sample as non-robust.
#     """
#     def optimize_model():
#         try:
#             model.optimize()
#         except gp.GurobiError as e:
#             print(f"Error optimizing Gurobi model: {e}")

#     # generate a thread to run optimize process
#     thread = threading.Thread(target=optimize_model)
#     thread.start()
    
#     # Wait for `timeout` seconds
#     thread.join(timeout)

#     # If thread is still running after timeout, stop and mark as non-robust
#     if thread.is_alive():
#         print(f"Optimization exceeded {timeout} seconds. Marking as non-robust.")
#         return False  
#     else:
#         return True  # Optimization completed in time

def test_robustness(model_dictionary, net, testloader, epsilon_input=1/255, epsilon_weight=2/255, epsilon_bias=2/255, epsilon_activation=2/255):
    time_exceed = 0
    worst_case_dis = 0
    total = 0
    for batch_idx,  (inputs, labels)  in enumerate(testloader, 0):
        start_time = time.time()
        inputs = inputs.view(inputs.size(0), -1)  # Flatten image to a vector
        labels = labels.item()
        #print(labels)
        #  Gurobi model
        model_gurobi = gp.Model("MLP_Robustness")

        # input variables
        x = model_gurobi.addVars(inputs.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        #print(inputs.shape[1])
        #  weight và bias variables (for layer 1-3)
        weight1 = model_gurobi.addVars(n_hidden_nodes, inputs.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight1")
        bias1 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias1")
        weight2 = model_gurobi.addVars(n_hidden_nodes, n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight2")
        bias2 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias2")
        weight3 = model_gurobi.addVars(n_hidden_nodes, n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight3")
        bias3 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias3")
        weight4 = model_gurobi.addVars(n_hidden_nodes, n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight4")
        bias4 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias4")
        weight5 = model_gurobi.addVars(n_hidden_nodes, n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight5")
        bias5 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias5")
        weight6 = model_gurobi.addVars(10, n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weight6")
        bias6 = model_gurobi.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias6")

        # input constraints
        for i in range(inputs.shape[1]):
            model_gurobi.addConstr(x[i] >= inputs[0][i].item() - epsilon_input, f"x_lower_{i}") #just 1 input each batch
            model_gurobi.addConstr(x[i] <= inputs[0][i].item() + epsilon_input, f"x_upper_{i}")
            #print(inputs[0][i].item())
        # weight constraints
        name = 'linear_layers'
        for i in range(n_hidden_nodes): 
            for j in range(inputs.shape[1]): 
                model_gurobi.addConstr(weight1[i,j] >= model_dictionary[name+'1weight'][i][j].item() - epsilon_weight, f"w1_lower_{i}_{j}")
                model_gurobi.addConstr(weight1[i,j] <= model_dictionary[name+'1weight'][i][j].item() + epsilon_weight, f"w1_upper_{i}_{j}") 
        for i in range(n_hidden_nodes): 
            for j in range(n_hidden_nodes): 
                model_gurobi.addConstr(weight2[i,j] >= model_dictionary[name+'2weight'][i][j].item() - epsilon_weight, f"w2_lower_{i}_{j}")
                model_gurobi.addConstr(weight2[i,j] <= model_dictionary[name+'2weight'][i][j].item() + epsilon_weight, f"w2_upper_{i}_{j}")
        for i in range(n_hidden_nodes): 
            for j in range(n_hidden_nodes): 
                model_gurobi.addConstr(weight3[i,j] >= model_dictionary[name+'3weight'][i][j].item() - epsilon_weight, f"w3_lower_{i}_{j}")
                model_gurobi.addConstr(weight3[i,j] <= model_dictionary[name+'3weight'][i][j].item() + epsilon_weight, f"w3_upper_{i}_{j}")
        for i in range(n_hidden_nodes): 
            for j in range(n_hidden_nodes): 
                model_gurobi.addConstr(weight4[i,j] >= model_dictionary[name+'4weight'][i][j].item() - epsilon_weight, f"w4_lower_{i}_{j}")
                model_gurobi.addConstr(weight4[i,j] <= model_dictionary[name+'4weight'][i][j].item() + epsilon_weight, f"w4_upper_{i}_{j}")
        for i in range(n_hidden_nodes): 
            for j in range(n_hidden_nodes): 
                model_gurobi.addConstr(weight5[i,j] >= model_dictionary[name+'5weight'][i][j].item() - epsilon_weight, f"w5_lower_{i}_{j}")
                model_gurobi.addConstr(weight5[i,j] <= model_dictionary[name+'5weight'][i][j].item() + epsilon_weight, f"w5_upper_{i}_{j}")
        for i in range(10): 
            for j in range(n_hidden_nodes): 
                model_gurobi.addConstr(weight6[i,j] >= model_dictionary[name+'6weight'][i][j].item() - epsilon_weight, f"w6_lower_{i}_{j}")
                model_gurobi.addConstr(weight6[i,j] <= model_dictionary[name+'6weight'][i][j].item() + epsilon_weight, f"w6_upper_{i}_{j}")
                #print(model_dictionary[name+'3weight'][i][j].item())
        #bias constraints 
        for i in range(n_hidden_nodes): 
            model_gurobi.addConstr(bias1[i] >= model_dictionary[name+'1bias'][i].item() - epsilon_bias, f"b1_lower_{i}")
            model_gurobi.addConstr(bias1[i] <= model_dictionary[name+'1bias'][i].item() + epsilon_bias, f"b1_upper_{i}")
        for i in range(n_hidden_nodes): 
            model_gurobi.addConstr(bias2[i] >= model_dictionary[name+'2bias'][i].item() - epsilon_bias, f"b2_lower_{i}")
            model_gurobi.addConstr(bias2[i] <= model_dictionary[name+'2bias'][i].item() + epsilon_bias, f"b2_upper_{i}")
        for i in range(n_hidden_nodes): 
            model_gurobi.addConstr(bias3[i] >= model_dictionary[name+'3bias'][i].item() - epsilon_bias, f"b3_lower_{i}")
            model_gurobi.addConstr(bias3[i] <= model_dictionary[name+'3bias'][i].item() + epsilon_bias, f"b3_upper_{i}")
        for i in range(n_hidden_nodes): 
            model_gurobi.addConstr(bias4[i] >= model_dictionary[name+'4bias'][i].item() - epsilon_bias, f"b4_lower_{i}")
            model_gurobi.addConstr(bias4[i] <= model_dictionary[name+'4bias'][i].item() + epsilon_bias, f"b4_upper_{i}")
        for i in range(n_hidden_nodes): 
            model_gurobi.addConstr(bias5[i] >= model_dictionary[name+'5bias'][i].item() - epsilon_bias, f"b5_lower_{i}")
            model_gurobi.addConstr(bias5[i] <= model_dictionary[name+'5bias'][i].item() + epsilon_bias, f"b5_upper_{i}")
        for i in range(10): 
            model_gurobi.addConstr(bias6[i] >= model_dictionary[name+'6bias'][i].item() - epsilon_bias, f"b6_lower_{i}")
            model_gurobi.addConstr(bias6[i] <= model_dictionary[name+'6bias'][i].item() + epsilon_bias, f"b6_upper_{i}")
        #print("----------------done constraints for input, weights, bias")
        
        # check loaded weights
        # print(f"Weight1[19]: {model_dictionary['linear_layers1weight'][19]}")
        # print(f"Bias1[19]: {model_dictionary['linear_layers1bias'][19].item()}")

        #define lower bound and upper bound of each linear layer output
        x_ub = inputs.to(device) + epsilon_input
        x_lb = inputs.to(device) - epsilon_input
        if device == 'cuda': 
            bounds = net.module.linear_bound(torch.cat([x_ub, x_lb], 0), epsilon_weight, epsilon_bias, epsilon_activation)
        else: 
            bounds = net.linear_bound(torch.cat([x_ub, x_lb], 0), epsilon_weight, epsilon_bias, epsilon_activation)        # check bounds
        #print(len(bounds))
        # for i, (ub, lb) in enumerate(bounds):
        #     print(f"Bound {i} upper_bound shape: {ub.shape}")
        #     print(f"Bound {i} lower_bound shape: {lb.shape}")
        #     upper_bound = bounds[i][0]  
        #     lower_bound = bounds[i][1] 
        #     print(bounds[i][0][0])

        # first layer (128 neurons)
        linear_output1 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput1") # define the linear output variables
        activation1 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="activation1") #relu output 
        x2 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="x2") # input of layer 2
        z1 = model_gurobi.addVars(n_hidden_nodes, vtype=GRB.BINARY, name="z1")  # binary variable for ReLU 
        upper_bound1 = bounds[0][0] # 128 upper bounds for output layer 1
        lower_bound1 = bounds[0][1] # 128 lower bounds for output layer 1
        for i in range(n_hidden_nodes):
            if (lower_bound1[0,i].item()>upper_bound1[0, i].item()): 
                print("***************CONFLICT*************")
            model_gurobi.addConstr(linear_output1[i] == gp.quicksum([x[j] * weight1[i, j] for j in range(inputs.shape[1])]) + bias1[i])

            model_gurobi.addConstr(activation1[i] <= linear_output1[i] - (1 - z1[i]) * lower_bound1[0,i].item())
            model_gurobi.addConstr(activation1[i] >= linear_output1[i])
            model_gurobi.addConstr(activation1[i] <= z1[i] * upper_bound1[0, i].item())
            model_gurobi.addConstr(activation1[i] >= 0)  

            #in case we want to use a linear bound to relax Relu
            #model_gurobi.addConstr(activation1[i] >= 0)
            #model_gurobi.addConstr(activation1[i] >= linear_output1[i])
            #model_gurobi.addConstr(activation1[i] <= upper_bound1[0, i]/(upper_bound1[0, i]-lower_bound1[0, i])*(linear_output1[i]-lower_bound1[0,i]))

            # print(" ------lower bound: ", lower_bound1[0,i].item())
            # print(" ------upper bound: ", upper_bound1[0,i].item())
        
            model_gurobi.addConstr(x2[i] >= activation1[i] - epsilon_activation, "x2_lower")
            model_gurobi.addConstr(x2[i] <= activation1[i] + epsilon_activation, "x2_upper")
            model_gurobi.addConstr(x2[i] >= 0)

        # layer 2 (128 neurons)
        linear_output2 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput2") # define the linear output variables
        activation2 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="activation2") #relu output
        x3 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="x3") # input of layer 3
        z2 = model_gurobi.addVars(n_hidden_nodes, vtype=GRB.BINARY, name="z2")  # binary variable for ReLU  
        upper_bound2 = bounds[1][0] # 128 upper bounds
        lower_bound2 = bounds[1][1] # 128 lower bounds 
        for i in range(n_hidden_nodes):
            model_gurobi.addConstr(linear_output2[i] == gp.quicksum([weight2[i, j] * x2[j] for j in range(n_hidden_nodes)]) + bias2[i])
            if (lower_bound2[0,i].item()>upper_bound2[0, i].item()): 
                print("***************CONFLICT*************")
            model_gurobi.addConstr(activation2[i] <= linear_output2[i] - (1 - z2[i]) * lower_bound2[0,i].item())
            model_gurobi.addConstr(activation2[i] >= linear_output2[i])  
            model_gurobi.addConstr(activation2[i] <= z2[i] * upper_bound2[0, i].item())
            model_gurobi.addConstr(activation2[i] >= 0)  

            # model_gurobi.addConstr(activation2[i] >= linear_output2[i])
            # model_gurobi.addConstr(activation2[i] <= upper_bound2[0, i]/(upper_bound2[0, i]-lower_bound2[0, i])*(linear_output2[i]-lower_bound2[0,i]))
            # model_gurobi.addConstr(activation2[i] >= 0)  
            
            model_gurobi.addConstr(x3[i] >= activation2[i] - epsilon_activation, "x3_lower")
            model_gurobi.addConstr(x3[i] <= activation2[i] + epsilon_activation, "x3_upper")
            model_gurobi.addConstr(x3[i] >= 0)

        #################################################
        # layer 3 (128 neurons)
        linear_output3 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput3") # define the linear output variables
        activation3 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="activation3") #relu output
        x4 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="x4") # input of layer 4
        z3 = model_gurobi.addVars(n_hidden_nodes, vtype=GRB.BINARY, name="z3")  # binary variable for ReLU  
        upper_bound3= bounds[2][0] # 128 upper bounds
        lower_bound3 = bounds[2][1] # 128 lower bounds 
        for i in range(n_hidden_nodes):
            model_gurobi.addConstr(linear_output3[i] == gp.quicksum([weight3[i, j] * x3[j] for j in range(n_hidden_nodes)]) + bias3[i])
            if (lower_bound3[0,i].item()>upper_bound3[0, i].item()): 
                print("***************CONFLICT*************")
            model_gurobi.addConstr(activation3[i] <= linear_output3[i] - (1 - z3[i]) * lower_bound3[0,i].item())
            model_gurobi.addConstr(activation3[i] >= linear_output3[i])  
            model_gurobi.addConstr(activation3[i] <= z3[i] * upper_bound3[0, i].item())
            model_gurobi.addConstr(activation3[i] >= 0)  

            # model_gurobi.addConstr(activation2[i] >= linear_output2[i])
            # model_gurobi.addConstr(activation2[i] <= upper_bound2[0, i]/(upper_bound2[0, i]-lower_bound2[0, i])*(linear_output2[i]-lower_bound2[0,i]))
            # model_gurobi.addConstr(activation2[i] >= 0)  
            
            model_gurobi.addConstr(x4[i] >= activation3[i] - epsilon_activation, "x4_lower")
            model_gurobi.addConstr(x4[i] <= activation3[i] + epsilon_activation, "x4_upper")
            model_gurobi.addConstr(x4[i] >= 0)

        ###########################################################
        # layer 4 (128 neurons)
        linear_output4 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput4") # define the linear output variables
        activation4 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="activation4") #relu output
        x5 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="x5") # input of layer 3
        z4 = model_gurobi.addVars(n_hidden_nodes, vtype=GRB.BINARY, name="z4")  # binary variable for ReLU  
        upper_bound4 = bounds[3][0] # 128 upper bounds
        lower_bound4 = bounds[3][1] # 128 lower bounds 
        for i in range(n_hidden_nodes):
            model_gurobi.addConstr(linear_output4[i] == gp.quicksum([weight4[i, j] * x4[j] for j in range(n_hidden_nodes)]) + bias4[i])
            if (lower_bound4[0,i].item()>upper_bound4[0, i].item()): 
                print("***************CONFLICT*************")
            model_gurobi.addConstr(activation4[i] <= linear_output4[i] - (1 - z4[i]) * lower_bound4[0,i].item())
            model_gurobi.addConstr(activation4[i] >= linear_output4[i])  
            model_gurobi.addConstr(activation4[i] <= z4[i] * upper_bound4[0, i].item())
            model_gurobi.addConstr(activation4[i] >= 0)  

            # model_gurobi.addConstr(activation2[i] >= linear_output2[i])
            # model_gurobi.addConstr(activation2[i] <= upper_bound2[0, i]/(upper_bound2[0, i]-lower_bound2[0, i])*(linear_output2[i]-lower_bound2[0,i]))
            # model_gurobi.addConstr(activation2[i] >= 0)  
            
            model_gurobi.addConstr(x5[i] >= activation4[i] - epsilon_activation, "x5_lower")
            model_gurobi.addConstr(x5[i] <= activation4[i] + epsilon_activation, "x5_upper")
            model_gurobi.addConstr(x5[i] >= 0)
        ###################################################
        # layer 5 (128 neurons)
        linear_output5 = model_gurobi.addVars(n_hidden_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="linearoutput5") # define the linear output variables
        activation5 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="activation5") #relu output
        x6 = model_gurobi.addVars(n_hidden_nodes, lb=0, ub=GRB.INFINITY, name="x6") # input of layer 3
        z5 = model_gurobi.addVars(n_hidden_nodes, vtype=GRB.BINARY, name="z5")  # binary variable for ReLU  
        upper_bound5 = bounds[4][0] # 128 upper bounds
        lower_bound5 = bounds[4][1] # 128 lower bounds 
        for i in range(n_hidden_nodes):
            model_gurobi.addConstr(linear_output5[i] == gp.quicksum([weight5[i, j] * x5[j] for j in range(n_hidden_nodes)]) + bias5[i])
            if (lower_bound5[0,i].item()>upper_bound5[0, i].item()): 
                print("***************CONFLICT*************")
            model_gurobi.addConstr(activation5[i] <= linear_output5[i] - (1 - z5[i]) * lower_bound5[0,i].item())
            model_gurobi.addConstr(activation5[i] >= linear_output5[i])  
            model_gurobi.addConstr(activation5[i] <= z5[i] * upper_bound5[0, i].item())
            model_gurobi.addConstr(activation5[i] >= 0)  
            # model_gurobi.addConstr(activation2[i] >= linear_output2[i])
            # model_gurobi.addConstr(activation2[i] <= upper_bound2[0, i]/(upper_bound2[0, i]-lower_bound2[0, i])*(linear_output2[i]-lower_bound2[0,i]))
            # model_gurobi.addConstr(activation2[i] >= 0)  
            
            model_gurobi.addConstr(x6[i] >= activation5[i] - epsilon_activation, "x6_lower")
            model_gurobi.addConstr(x6[i] <= activation5[i] + epsilon_activation, "x6_upper")
            model_gurobi.addConstr(x6[i] >= 0)

        ###########################################################
        # output layer (10 neurons)
        output = model_gurobi.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="output")
        upper_bound6 = bounds[5][0] # 128 upper bounds
        lower_bound6 = bounds[5][1] # 128 lower bounds 
        for i in range(10):
            model_gurobi.addConstr(output[i] == gp.quicksum([weight6[i, j] * x6[j] for j in range(n_hidden_nodes)]) + bias6[i])
            model_gurobi.addConstr(output[i] >= lower_bound6[0,i].item())
            # print("----Output lower bound:", lower_bound3[0,i].item())
            # print("----Output upper bound:", upper_bound3[0,i].item())
            model_gurobi.addConstr(output[i] <= upper_bound6[0,i].item())

        # orig_outputs = net(torch.cat([inputs.to(device),inputs.to(device)], 0))
        # value_fl_batch = orig_outputs[:orig_outputs .shape[0]//2]
        # value_fl_batch = F.softmax(value_fl_batch, dim=1)
        # #print(value_fl_batch.shape)
        # orig_output = value_fl_batch[0][labels].item() # output of true class


        num_classes = 10
        min_output = []
        max_output = []
        # Find the maximum value of each logit
        for i in range(num_classes): 
            M = model_gurobi.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"M_class_{i}")
            model_gurobi.addConstr(M == output[i])
            model_gurobi.setObjective(M, GRB.MAXIMIZE)
            model_gurobi.optimize()
            # Check if the optimization was successful
            if model_gurobi.status == GRB.OPTIMAL:
                # Retrieve the optimized value of M
                print(f'***************************Class {i+1} max value: {M.X}')
                max_output.append(M.X)
            elif model_gurobi.status == GRB.INFEASIBLE:
                # Handle infeasibility
                print(f"Model is infeasible for class {i}. Computing IIS...")
                model_gurobi.computeIIS()
                model_gurobi.write(f"model_class_{i}_infeasible.ilp")

        # Find the minimum value for each logit
        for i in range(num_classes): 
            m = model_gurobi.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"m_class_{i}")
            model_gurobi.addConstr(m == output[i])
            model_gurobi.setObjective(m, GRB.MINIMIZE)
            model_gurobi.optimize()
            
            # Check if the optimization was successful
            if model_gurobi.status == GRB.OPTIMAL:
                # Retrieve the optimized value of m
                print(f'***************************Class {i+1} min value: {m.X}')
                min_output.append(m.X)
            elif model_gurobi.status == GRB.INFEASIBLE:
                # Handle infeasibility
                print(f"Model is infeasible for class {i}. Computing IIS...")
                model_gurobi.computeIIS()
                model_gurobi.write(f"model_class_{i}_infeasible.ilp")
        
        # result = {'Min ': min_output, 'Max ': max_output }
        # path = f'opt_results/sample{batch_idx+1}.xlsx'
        # DictExcelSaver.save(result,path)
             
        min_output = torch.tensor(min_output).unsqueeze(0)  # Shape: (1, 10)
        max_output = torch.tensor(max_output).unsqueeze(0)  # Shape: (1, 10)

        best_case = min_output.clone() 
        best_case[0, labels] = max_output[0, labels]  # Replace the true class logit with the max value
        best = (F.softmax(best_case, dim=1))[0, labels].item()  # Compute softmax and get value for true class

        worst_case = max_output.clone()
        worst_case[0, labels] = min_output[0, labels]  # Replace the true class logit with the min value
        worst = (F.softmax(worst_case, dim=1))[0, labels].item()  # Compute softmax and get value for true class

        # diff =  max(best - orig_output, orig_output - worst)
        # #diff =  max(outputs_ub[0,targets] - orig_output, orig_output - outputs_lb[0,targets])
        # if diff > worst_case_dis: 
        #     worst_case_dis = diff
        result = {'Best ': best, 'Worst ': worst }
        path = f'opt_results/exp16_sample{batch_idx+1}.xlsx'
        DictExcelSaver.save(result,path)
        total += 1
        end_time = time.time()
        execution_time = end_time - start_time  
        print(f"Processed sample {batch_idx} in {execution_time:.2f} seconds.")
    #     print(f'---------------done {total} samples, worst case discrepancy found is {worst_case_dis}')    

    # print(f"*************Worst case discrepancy with ep_i = {epsilon_input}, ep_w = {epsilon_weight}, ep_b = {epsilon_bias},ep_a = {epsilon_activation}: {worst_case_dis}")
    # result = {'ep_i':epsilon_input, 'ep_w': epsilon_weight, 'ep_b': epsilon_bias, 'ep_a':epsilon_activation, 'Worst case discrepancy': worst_case_dis }
    # path = f'opt_results/exp16.xlsx'
    # DictExcelSaver.save(result,path)
if __name__ == '__main__':
    test_robustness(model_dictionary,net, testloader, epsilon_input=1/1023, epsilon_weight=1/1023, epsilon_bias=1/1023, epsilon_activation=1/1023)
