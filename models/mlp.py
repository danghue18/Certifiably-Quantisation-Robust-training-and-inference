from torch import nn
import torch
from torchsummary import summary

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3


class MLP(nn.Module):
    def __init__(self, n_hidden_nodes_1=128, n_hidden_nodes_2=128):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(28 * 28 * 1, n_hidden_nodes_1),
          nn.ReLU(),
          nn.Linear(n_hidden_nodes_1, n_hidden_nodes_2),
          nn.ReLU(),
        #   nn.Linear(n_hidden_nodes_2, n_hidden_nodes_3),
        #   nn.ReLU(),
          nn.Linear(n_hidden_nodes_2, 10)
        )

    def forward(self, x):
        return self.layers(x)


def mlp():
    return MLP(128,64)


def test():
    net = mlp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)  # Move the model to CUDA if available
    # x = torch.randn(28,28)
    # y = net(x)
    #print(net)
    summary(net, (28, 28))


    # Print the state dictionary of the model
    print("Model State Dictionary:")
    for param_tensor in net.state_dict():
        print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")


if __name__=="__main__":
    test()
