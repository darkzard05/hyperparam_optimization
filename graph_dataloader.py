import torch
from typing import Tuple

from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T


def preprocess_data(data, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (data.x - data.x.mean()) / data.x.std()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    return x.to(device), edge_index.to(device), edge_attr.to(device)


def get_dataset(path, name, transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
        
    return Planetoid(root=path, name=name, transform=transform)


def get_dataloader(data, num_neighbors, batch_size, num_workers):
    kwargs = {'num_workers': num_workers, 'pin_memory': True,
              'persistent_workers': True if num_workers > 0 else False,
              'shuffle': True}
    train_loader = NeighborLoader(data=data,
                                  num_neighbors=num_neighbors,
                                  input_nodes=data.train_mask,
                                  batch_size=batch_size,
                                  **kwargs)
    val_loader = NeighborLoader(data=data,
                                num_neighbors=num_neighbors,
                                input_nodes=data.val_mask,
                                batch_size=1,
                                **kwargs)
    test_loader = NeighborLoader(data=data,
                                 num_neighbors=num_neighbors,
                                 input_nodes=data.test_mask,
                                 batch_size=1,
                                 **kwargs)
    return train_loader, val_loader, test_loader