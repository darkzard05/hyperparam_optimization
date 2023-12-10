import torch
from typing import Tuple

from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T


def preprocess_data(data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (data.x - data.x.mean(dim=0, keepdim=True)) / data.x.std(dim=0, keepdim=True)
    return x, data.edge_index, data.edge_attr


def get_dataset(path, name, transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
        
    return Planetoid(root=path, name=name, transform=transform)


def get_dataloader(data, num_neighbors, batch_size, num_workers):
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': True,
                     'persistent_workers': True if num_workers > 0 else False,
                     'shuffle': True}
    
    def create_loader(mask):
        return NeighborLoader(data=data,
                              num_neighbors=num_neighbors,
                              input_nodes=mask,
                              batch_size=batch_size,
                              **loader_kwargs)
    
    train_loader = create_loader(data.train_mask)
    val_loader = create_loader(data.val_mask)
    test_loader = create_loader(data.test_mask)
    
    return train_loader, val_loader, test_loader