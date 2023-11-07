from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T

def get_dataset(path, name, transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
        
    return Planetoid(root=path, name=name, transform=transform)

def get_train_loader(data, num_neighbors, batch_size, num_workers=4):
    kwargs = {'pin_memory': True, 'shuffle': True}
    train_loader = NeighborLoader(data=data, input_nodes=data.train_mask,
                                  num_neighbors=num_neighbors, batch_size=batch_size,
                                  num_workers=num_workers, **kwargs)
    return train_loader