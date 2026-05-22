from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_path

from graph_mixup.augmentations.basic.typing import DropPathDatasetMethodConfig


class DropPath(BaseTransform):
    def __init__(self, params: DropPathDatasetMethodConfig) -> None:
        self.p = params.p
        self.walks_per_node = params.walks_per_node
        self.walk_length = params.walk_length

    def forward(self, data: Data) -> Data:
        edge_index, edge_mask = dropout_path(
            data.edge_index, self.p, self.walks_per_node, self.walk_length
        )
        data.edge_index = edge_index

        if "edge_attr" in data.keys():
            data.edge_attr = data.edge_attr[edge_mask]

        return data
