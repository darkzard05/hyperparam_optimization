import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv
from torch_geometric.typing import Adj, OptTensor


class appnp(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 dropout: float = 0.5,
                 K: int = 50,
                 alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.lin = Linear(in_channels, out_channels)
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


class splineconv(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 dropout: float = 0.5,
                 kernel_size: int = 2,
                 n_units: int = 32):
        super().__init__()
        self.n_units = n_units
        self.dropout = dropout
        self.conv_1 = SplineConv(in_channels, self.n_units,
                                 dim=1, kernel_size=kernel_size)
        self.conv_2 = SplineConv(self.n_units, out_channels,
                                 dim=1, kernel_size=kernel_size)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

    
class gat(torch.nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 dropout: float = 0.5,
                 n_units: int = 32,
                 heads: int = 8):
        super().__init__()
        self.dropout = dropout
        self.n_units = n_units
        self.heads = heads
        self.conv_1 = GATConv(in_channels, self.n_units, heads=self.heads,
                              dropout=self.dropout)
        self.conv_2 = GATConv(self.n_units * self.heads, out_channels,
                              dropout=self.dropout)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self,
                x: Tensor, edge_index: Adj, edge_attr : OptTensor = None) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)