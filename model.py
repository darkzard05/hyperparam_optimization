import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, APPNP, SplineConv

class appnp(torch.nn.Module):
    def __init__(self, dataset, trial):
        super().__init__()
        self.num_layers = trial.suggest_int('num_layers', 1, 3)
        self.K = trial.suggest_int('K', 1, 200)
        self.alpha = trial.suggest_float('alpha', 0, 0.2)
        self.units_list = []
        if self.num_layers >= 2:
            for epoch in range(self.num_layers-1):
                self.units_list.append(trial.suggest_int(f'n_units_{epoch}', 5, 200))
        self.layers_list = nn.ModuleList()
        self.dropout_list = []
        in_features = dataset[0].num_features
        for epoch in range(self.num_layers):
            self.dropout_list.append(trial.suggest_float(f'dropout_{epoch}', 0.2, 0.5))
            if epoch == self.num_layers-1:
                out_features = dataset.num_classes
            else:
                out_features = self.units_list[epoch]
            self.layers_list.append(Linear(in_features, out_features))
            in_features = out_features
        self.prop = APPNP(K=self.K, alpha=self.alpha)
        
    def reset_parameters(self):
        for layer in self.layers_list:
            layer.reset_parameters()
        
    def forward(self, d):
        x, edge_index = d.x, d.edge_index
        for i, layer in enumerate(self.layers_list):
            # x = F.dropout(x, training=self.training)
            x = F.dropout(x, p=self.dropout_list[i], training=self.training)
            x = layer(x).relu() 
        x = self.prop(x, edge_index)
        return x

class splineconv(torch.nn.Module):
    def __init__(self, dataset, trial):
        super().__init__()
        self.num_layers = trial.suggest_int('num_layers', 1, 3)
        self.units_list = []
        if self.num_layers >= 2:
            for n in range(self.num_layers -1):
                self.units_list.append(trial.suggest_int(f'n_units_{n}', 4, 128))
        self.layers_list = nn.ModuleList()
        self.dropout_list = []
        in_features = dataset[0].num_features
        for epoch in range(self.num_layers):
            self.dropout_list.append(trial.suggest_float(f'dropout_{epoch}', 0.2, 0.5))
            if epoch == self.num_layers-1:
                out_features = dataset.num_classes
            else:
                out_features = self.units_list[epoch]
            self.layers_list.append(SplineConv(in_features,
                                               out_features,
                                               dim = 1,
                                               kernel_size = trial.suggest_int('kernel_size', 1, 6)))
            in_features = out_features

    def reset_parameters(self):
        for layer in self.layers_list:
            layer.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers_list):
            x = F.dropout(x, p=self.dropout_list[i], training=self.training)
            x = layer(x, edge_index, edge_attr).relu()
        return F.log_softmax(x, dim=1)