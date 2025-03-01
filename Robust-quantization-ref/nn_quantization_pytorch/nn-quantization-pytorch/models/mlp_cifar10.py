import torch.nn as nn

class CIFAR10_MLP(nn.Module):
    def __init__(self):
        super(CIFAR10_MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
        self.activation = nn.ReLU()  
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x

# def MLP_cifar10(): 
#     model = CIFAR10_MLP()
#     return model 
import os
import torch
def cifar10_mlp(pretrained=False, weight_path=None, **kwargs):
    """
    MLP architecture for CIFAR-10 classification.

    Args:
        pretrained (bool): If True, loads pretrained weights from a local file.
        weight_path (str): Path to the local file containing pretrained weights.
        **kwargs: Additional arguments to customize the MLP model.
    """
    model = CIFAR10_MLP(**kwargs)
    if pretrained:
        if weight_path is None:
            raise ValueError("Please specify the `weight_path` for loading pretrained weights.")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"The specified weight file does not exist: {weight_path}")
        
        # Load the state_dict from the local file
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {weight_path}")
    return model

