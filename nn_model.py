import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv

class appnp(torch.nn.Module):
    def __init__(self, dataset, K, alpha, dropout):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        
        self.lin = Linear(dataset[0].num_features, dataset.num_classes)
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

class splineconv(torch.nn.Module):
    def __init__(self, dataset, kernel_size, n_units, dropout):
        super().__init__()
        self.n_units = n_units
        self.dropout = dropout
        
        self.conv_1 = SplineConv(-1, self.n_units,
                                 dim=1, kernel_size=kernel_size)
        self.conv_2 = SplineConv(self.n_units, dataset.num_classes,
                                 dim=1, kernel_size=kernel_size)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)
    
class gat(torch.nn.Module):
    def __init__(self, dataset, n_units, heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.n_units = n_units
        self.heads = heads
        
        self.conv_1 = GATConv(-1, self.n_units, heads=self.heads,
                              dropout=self.dropout)
        self.conv_2 = GATConv(self.n_units * self.heads, dataset.num_classes,
                              dropout=self.dropout)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)