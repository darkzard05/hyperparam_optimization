import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv, GATConv

class appnp(torch.nn.Module):
    def __init__(self, dataset, trial):
        super().__init__()
        self.in_features = dataset[0].num_features
        self.out_features = dataset.num_classes
        
        self.K = trial.suggest_int('K', 5, 200)
        self.alpha = trial.suggest_float('alpha', 0.05, 0.2)
        self.dropout = trial.suggest_float('dropout', 0.0, 0.7)
        
        self.lin = Linear(self.in_features, self.out_features)
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
    def __init__(self, dataset, trial):
        super().__init__()
        self.in_features = dataset[0].num_features
        self.out_features = dataset.num_classes
        
        self.dropout = trial.suggest_float('dropout', 0.0, 0.7)
        self.n_units = trial.suggest_categorical('n_units', [2 ** i for i in range(2, 10)])
        
        self.conv_1 = SplineConv(self.in_features, self.n_units, dim=1, kernel_size=2)
        self.conv_2 = SplineConv(self.n_units, self.out_features, dim=1, kernel_size=2)
        
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
    def __init__(self, dataset, trial):
        super().__init__()
        self.in_features = dataset[0].num_features
        self.out_features = dataset.num_classes
        
        self.dropout = trial.suggest_float('dropout', 0.0, 0.7)
        self.n_units = trial.suggest_categorical('n_units', [2**i for i in range(2, 8)])
        self.heads = trial.suggest_int('heads', 1, 8)
        
        self.conv_1 = GATConv(self.in_features, self.n_units, heads=self.heads,
                              dropout=self.dropout)
        self.conv_2 = GATConv(self.n_units * self.heads, self.out_features,
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