from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T

def load_dataset(path, name, split='public', transform=T.TargetIndegree()):
    if name == 'Reddit':
        return Reddit(root=path, transform=transform)
    else:
        return Planetoid(root=path, name=name, split=split, transform=transform)