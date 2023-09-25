import torch
from torch import Tensor
from torch.nn import ModuleList
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
                 in_channels: int,
                 out_channels: int,
                 n_units: int,
                 K: int = 50,
                 alpha: float = 0.1,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 num_layers: int = 1
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_units = n_units
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.num_layers = num_layers
        self.model_list = ModuleList()
        if self.num_layers == 1:
            self.model_list = ModuleList([Linear(self.in_channels, self.out_channels)])
        elif self.num_layers == 2:
            self.model_list = ModuleList([Linear(self.in_channels, self.n_units),
                                          Linear(self.n_units, self.out_channels)])
        else:
            self.model_list.append(Linear(self.in_channels, self.n_units))
            for _ in range(2, self.num_layers):
                self.model_list.append(Linear(self.n_units, self.n_units))
            self.model_list.append(Linear(self.n_units, self.out_channels))
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        self.reset_parameters()
        
    def reset_parameters(self):
        for linear in self.model_list:
            linear.reset_parameters()
        
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


class SplineconvModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_units: int = 32,
                 kernel_size: int = 2,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 num_layers: int = 1
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_units = n_units
        self.dropout = dropout
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.num_layers = num_layers
        self.model_list = ModuleList()
        if self.num_layers == 1:
            self.model_list = ModuleList([SplineConv(self.in_channels, self.out_channels,
                                                  dim=1, kernel_size=self.kernel_size)])
        elif self.num_layers == 2:
            self.model_list = ModuleList([SplineConv(self.in_channels, self.n_units,
                                                  dim=1, kernel_size=self.kernel_size),
                                          SplineConv(self.n_units, self.out_channels,
                                                  dim=1, kernel_size=self.kernel_size)])
        else:
            self.model_list.append(SplineConv(self.in_channels, self.n_units,
                                           dim=1, kernel_size=self.kernel_size))
            for _ in range(2, self.num_layers):
                self.model_list.append(SplineConv(self.n_units, self.n_units,
                                               dim=1, kernel_size=self.kernel_size))
            self.model_list.append(SplineConv(self.n_units, self.out_channels,
                                           dim=1, kernel_size=self.kernel_size))
        # self.conv_1 = SplineConv(self.in_channels, self.n_units,
        #                          dim=1, kernel_size=self.kernel_size)
        # self.conv_2 = SplineConv(self.n_units, self.out_channels,
        #                          dim=1, kernel_size=self.kernel_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        for splineconv in self.model_list:
            splineconv.reset_parameters()
        
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for splineconv in self.model_list:
            x = splineconv(x, edge_index, edge_attr)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    
class GATModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 n_units: int = 32,
                 heads: int = 8,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 num_layers: int = 1
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_units = n_units
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.model_list = ModuleList()
        if self.num_layers == 1:
            self.model_list = ModuleList([GATConv(self.in_channels, self.out_channels, dropout=self.dropout)])
        elif self.num_layers == 2:
            self.model_list = ModuleList([GATConv(self.in_channels, self.n_units, heads=self.heads,
                                                  dropout=self.dropout),
                                          GATConv(self.n_units * self.heads, self.out_channels,
                                                  dropout=self.dropout)])
        else:
            self.model_list.append(GATConv(self.in_channels, self.n_units, heads=self.heads,
                                           dropout=self.dropout))
            temp = self.n_units * self.heads
            for _ in range(2, self.num_layers):
                self.model_list.append(GATConv(temp, temp, heads=self.heads,
                                               dropout=self.dropout))
                temp *= self.heads
            self.model_list.append(GATConv(temp, self.out_channels,
                                           dropout=self.dropout))
        self.reset_parameters()
        
    def reset_parameters(self):
        for gatconv in self.model_list:
            gatconv.reset_parameters()
        
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr : OptTensor = None) -> Tensor:
        for gatconv in self.model_list:
            x = gatconv(x, edge_index, edge_attr)
            x = self.activation(x)
        # x = self.conv_1(x, edge_index, edge_attr)
        # x = self.activation(x)
        # x = self.conv_2(x, edge_index, edge_attr)
        return x