from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import NeighborSampler
import torch_geometric.transforms as T

def load_dataset(path, name, split='public', transform=T.TargetIndegree()):
    if name == 'Reddit':
        dataset = Reddit(root=path, transform=transform)
        data = dataset[0]
        loader = NeighborSampler(edge_index=data.edge_index,
                                 sizes=[20, 10],
                                 batch_size=128)
        return loader
    else:
        return Planetoid(root=path, name=name, split=split, transform=transform)