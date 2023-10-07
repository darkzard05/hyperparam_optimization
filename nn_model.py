import torch
from torch import Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv
from torch_geometric.typing import Adj, OptTensor


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU(),
    'leakyrelu': torch.nn.LeakyReLU(),
    'elu': torch.nn.ELU()
}


def get_activation(activation):
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f'Activation {activation} is not a supported activation function. Supported activation: {", ".join(ACTIVATION_FUNCTIONS.keys())}' )
    return ACTIVATION_FUNCTIONS[activation]


def build_layers(model, in_channels, out_channels, n_units, num_layers,
                     heads=1, multi_factor=1, **kwargs):
    model_list = ModuleList()
    for layer in range(1, num_layers+1):
        if layer == num_layers:
            current = out_channels
        else:
            current = n_units * multi_factor
        model_list.append(model(in_channels, current, **kwargs))
        in_channels = current
    return model_list


class BaseModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_units: int = 16,
                 dropout: float = 0.5,
                 activation: str = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_units = n_units
        self.dropout = dropout
        self.activation = get_activation(activation)
        
    def reset_parameters(self):
        for layer in self.model_list:
            layer.reset_parameters()


class APPNPModel(BaseModel):
    def __init__(self,
                 *args,
                 K: int = 50,
                 alpha: float = 0.1,
                 num_layers: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.alpha = alpha
        
        self.num_layers = num_layers
        self.model_list = build_layers(Linear, self.in_channels, self.out_channels,
                                           self.n_units, self.num_layers)
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        self.reset_parameters()
    
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


class SplineconvModel(BaseModel):
    def __init__(self,
                 *args,
                 kernel_size: int = 2,
                 num_layers: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size

        self.num_layers = num_layers
        self.model_list = build_layers(SplineConv, self.in_channels, self.out_channels,
                                           self.n_units, self.num_layers,
                                           dim=1, kernel_size=self.kernel_size)
        self.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for splineconv in self.model_list:
            x = splineconv(x, edge_index, edge_attr)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    
class GATModel(BaseModel):
    def __init__(self,
                 *args,
                 heads: int = 8,
                 num_layers: int = 1,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.heads = heads
        
        self.num_layers = num_layers
        self.model_list = build_layers(GATConv, self.in_channels, self.out_channels,
                                           self.n_units, self.num_layers, heads=self.heads,
                                           dropout=self.dropout, multi_factor=self.heads)
        self.reset_parameters()
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for i, gatconv in enumerate(self.model_list):
            x = gatconv(x, edge_index, edge_attr)
            if i != len(self.model_list)-1:
                x = self.activation(x)
        return x