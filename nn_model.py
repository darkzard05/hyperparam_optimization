import torch
from torch import nn, Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv
from torch_geometric.typing import Adj, OptTensor


ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.2),
    'I': nn.LeakyReLU(negative_slope=1),
    'elu': nn.ELU(),
    'tanh': nn.Tanh(),
    'prelu': nn.PReLU()
}


def get_activation(activation):
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f'Activation {activation} is not a supported activation function. Supported activation: {", ".join(ACTIVATION_FUNCTIONS.keys())}' )
    return ACTIVATION_FUNCTIONS[activation]


def build_multi_layers(model, in_channels, out_channels, num_layers, n_units,
                       **kwargs):
    model_list = ModuleList()
    if num_layers == 1:
        return ModuleList([model(in_channels, out_channels, **kwargs)])
    else:
        model_list.append(model(in_channels, n_units[0], **kwargs))
    for i in range(1, num_layers-1):
        if i == num_layers-2:
            model_list.append(model(n_units[i-1], out_channels, **kwargs))
        else:
            model_list.append(model(n_units[i-1], n_units[i], **kwargs))
    return model_list


class APPNPModel(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.model_list = build_multi_layers(model=Linear, in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             num_layers=self.num_layers, n_units=self.n_units)
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        self.activation = get_activation(self.activation)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.model_list:
            layer.reset_parameters()
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for linear in self.model_list:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = linear(x)
            x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return x


class SplineconvModel(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.model_list = build_multi_layers(SplineConv, self.in_channels, self.out_channels,
                                             self.num_layers, self.n_units,
                                             dim=1, kernel_size=self.kernel_size)
        self.activation = get_activation(self.activation)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model_list:
            layer.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for conv in self.model_list:
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)
        return x

    
class GATModel(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.model_list = build_multi_layers(GATConv, self.in_channels, self.out_channels,
                                             self.n_units, self.num_layers,
                                             heads=self.heads,
                                             dropout=self.dropout)
        self.activation = get_activation(self.activation)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.model_list:
            layer.reset_parameters()
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        for i, conv in enumerate(self.model_list):
            x = conv(x, edge_index, edge_attr)
            if i != len(self.model_list)-1:
                x = self.activation(x)
        return x