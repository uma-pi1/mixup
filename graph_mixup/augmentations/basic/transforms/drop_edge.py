from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge

from graph_mixup.augmentations.basic.typing import DropEdgeDatasetMethodConfig


class DropEdge(BaseTransform):
    def __init__(self, params: DropEdgeDatasetMethodConfig) -> None:
        self.p = params.p

    def forward(self, data: Data) -> Data:
        edge_index, edge_mask = dropout_edge(data.edge_index, self.p)
        data.edge_index = edge_index

        if "edge_attr" in data.keys():
            data.edge_attr = data.edge_attr[edge_mask]

        return data
