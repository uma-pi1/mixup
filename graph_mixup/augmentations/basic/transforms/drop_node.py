from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_node

from graph_mixup.augmentations.basic.typing import DropNodeDatasetMethodConfig


class DropNode(BaseTransform):
    def __init__(self, params: DropNodeDatasetMethodConfig) -> None:
        self.p = params.p

    def forward(self, data: Data) -> Data:
        while True:
            # Check if graph still has at least one node left.
            # If not => resample.

            edge_index, edge_mask, node_mask = dropout_node(
                data.edge_index, self.p, relabel_nodes=True
            )
            if node_mask.sum().item() > 0:
                break

        data.edge_index = edge_index
        data.x = data.x[node_mask]

        if "edge_attr" in data.keys():
            data.edge_attr = data.edge_attr[edge_mask]

        if "num_nodes" in data.keys():
            data.num_nodes = node_mask.sum().item()

        return data
