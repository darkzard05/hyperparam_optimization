import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv

class appnp(torch.nn.Module):
    def __init__(self, dataset, trial):
        super().__init__()
        self.num_layers = trial.suggest_int('num_layers', 1, 3)
        self.K = trial.suggest_int('K', 5, 200)
        self.alpha = trial.suggest_float('alpha', 0.0, 0.5)
        
        self.layers_list = nn.ModuleList()
        self.dropout_list = []
        self.units_list = []
        
        if self.num_layers >= 2:
            for epoch in range(1, self.num_layers):
                self.units_list.append(trial.suggest_categorical(f'n_units_{epoch}',
                                                                 [4, 8, 16, 32, 64, 128, 256, 512]))
        
        self.in_features = dataset[0].num_features
        
        for epoch in range(1, self.num_layers+1):
            self.dropout_list.append(trial.suggest_float(f'dropout_{epoch}', 0.0, 1.0))
            if epoch == self.num_layers:
                out_features = dataset.num_classes
            else:
                out_features = self.units_list[epoch-1]
            self.layers_list.append(Linear(self.in_features, out_features))
            self.in_features = out_features
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        
    def reset_parameters(self):
        for layer in self.layers_list:
            layer.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers_list):
            x = F.dropout(x, p=self.dropout_list[i], training=self.training)
            x = layer(x).relu() 
        x = self.prop(x, edge_index)
        return x

class splineconv(torch.nn.Module):
    def __init__(self, dataset, trial):
        super().__init__()
        self.in_features = dataset[0].num_features
        self.out_features = dataset.num_classes
        self.dropout_1 = trial.suggest_float('dropout_1', 0.0, 1.0)
        self.dropout_2 = trial.suggest_float('dropout_2', 0.0, 1.0)
        self.n_units = trial.suggest_categorical('n_units', [4, 8, 16, 32, 64, 128, 256, 512, 1024])
        self.conv_1 = SplineConv(self.in_features, self.n_units, dim=1, kernel_size=2)
        self.conv_2 = SplineConv(self.n_units, self.out_features, dim=1, kernel_size=2)
        
    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout_1, training=self.training)
        x = self.conv_1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=self.dropout_2, training=self.training)
        x = self.conv_2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)