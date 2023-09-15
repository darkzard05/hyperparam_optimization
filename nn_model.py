import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv
from torch_geometric.typing import Adj, OptTensor


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'leakyrelu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU
}


class APPNPModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 K: int = 50,
                 alpha: float = 0.1,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.lin = Linear(in_channels, out_channels)
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = self.lin(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return x


class SplineconvModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int = 2,
                 n_units: int = 32,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.n_units = n_units
        self.dropout = dropout
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.conv_1 = SplineConv(in_channels, self.n_units,
                                 dim=1, kernel_size=self.kernel_size)
        self.conv_2 = SplineConv(self.n_units, out_channels,
                                 dim=1, kernel_size=self.kernel_size)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = self.conv_1(x, edge_index, edge_attr)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return x

    
class GATModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 n_units: int = 32,
                 heads: int = 8,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.heads = heads
        self.dropout = dropout
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.conv_1 = GATConv(in_channels, self.n_units, heads=self.heads,
                              dropout=self.dropout)
        self.conv_2 = GATConv(self.n_units * self.heads, out_channels,
                              dropout=self.dropout)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = self.conv_1(x, edge_index, edge_attr)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return x