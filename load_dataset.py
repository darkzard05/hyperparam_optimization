from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T

def load_dataset(path, name,
                 split='public',
                 batch_size=64,
                 transform=T.TargetIndegree()):
    if name == 'Reddit':
        dataset = Reddit(root=path, transform=transform)
        return DataLoader(dataset, batch_size=batch_size)
    
    return Planetoid(root=path, name=name, split=split, transform=transform)