from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch_geometric.data import Data
import torch


class NormalizedDegree(BaseTransform):
    def __init__(self, mean: float, std: float) -> None:
        assert std > 0, "standard deviation must be positive"
        self.mean = mean
        self.std = std

    def forward(self, data: Data) -> Data:
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
