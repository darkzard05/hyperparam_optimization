import torch
from typing import Tuple

from torch_geometric.loader import DataLoader, NodeLoader, NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T

from optuna.samplers import BaseSampler


def preprocess_data(data, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (data.x - data.x.mean()) / data.x.std()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    return x.to(device), edge_index.to(device), edge_attr.to(device)


def get_dataset(path, name, transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
        
    return Planetoid(root=path, name=name, transform=transform)


def get_train_loader(data, num_neighbors, batch_size, num_workers):
    kwargs = {'num_workers': num_workers, 'pin_memory': True,
              'persistent_workers': True, 'shuffle': True}
    train_loader = NeighborLoader(data=data,
                                  num_neighbors=num_neighbors,
                                  input_nodes=data.train_mask,
                                  batch_size=batch_size,
                                  **kwargs)
    return train_loader