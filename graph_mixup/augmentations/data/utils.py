import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform, Compose, Constant
from torch_geometric.utils import degree

from graph_mixup.transforms.one_hot_label_transform import OneHotLabel


def initialize_transforms(num_classes: int, num_features: int) -> BaseTransform:
    transforms: list[BaseTransform] = [OneHotLabel(num_classes)]

    if num_features == 0:
        transforms.append(Constant())

    return Compose(transforms)


def compute_max_degree_mean_std(
    dataset: TUDataset | list[Data],
) -> tuple[int, float, float]:
    max_degree = 0
    all_degrees = []
    for data in dataset:
        all_degrees += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, all_degrees[-1].max().item())

    deg = torch.cat(all_degrees, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()
    return max_degree, mean, std
