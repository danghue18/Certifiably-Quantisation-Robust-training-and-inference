import torch
from utils import progress_bar
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def print_accuracy(net, trainloader, testloader, device, test=True, ep_i = 0, ep_w = 0, ep_b = 0, ep_a = 0):
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
            x_ub = images + ep_i
            x_lb = images - ep_i
            
            outputs = net(torch.cat([x_ub,x_lb], 0), ep_w, ep_b, ep_a)
            
            z_ub = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            #print(z_lb==z_ub)
            #loss_nor = criterion(z_ub, labels)
            lb_mask = torch.eye(10).to(device)[labels]
            ub_mask = 1 - lb_mask
            outputs = z_lb * lb_mask + z_ub * ub_mask
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

    print('Accuracy of the network on the', total, loadertype, 'images: ',correct, 'with epsilon input = ', ep_i,' epsilon param = ', ep_w, ' epsilon activation = ', ep_a )
    return correct, loss_aver
