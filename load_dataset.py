from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T

def load_dataset(path, name,
                 split='public',
                 transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
        
    return Planetoid(root=path, name=name, split=split, transform=transform)

def load_train_loader(data, num_neighbors, batch_size):
    train_loader = NeighborLoader(data=data, num_neighbors=num_neighbors,
                                  input_nodes=data.train_mask, batch_size=batch_size)
    return train_loader