import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Imports import device


class NN(nn.Module):
    def __init__(self, layer_dims, activation=nn.ReLU, dropout_p=0.0):
        super(NN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            if i < len(self.layers) - 1:  # No activation or dropout after the last layer
                x = self.dropout(x)
        
        # return 50 * torch.tanh(x.squeeze(1))
        return x.squeeze(1)

class SNN(nn.Module):
    def __init__(self, layer_dims, activation=nn.ReLU, dropout_p=0.0):
        super(SNN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            if i < len(self.layers) - 1:  # No dropout after the last layer
                x = self.dropout(x)
        
        return x.squeeze(1)


def is_valid_sigma_torch(t_sigma):
    return ((t_sigma >= 0) & (t_sigma <= 1) & (~t_sigma.isnan())).all()


def init_weights_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# if __name__ == "__main__":

#     nn_alx = NN(4*52+4,10,1).to(device).apply(init_weights_zero)
#     nn_bob = NN(4*52+4,10,1).to(device).apply(init_weights_zero)


#     torch.save(nn_alx, 'nn_alx.pth')
#     torch.save(nn_bob, 'nn_bob.pth')
#     import pdb; pdb.set_trace()