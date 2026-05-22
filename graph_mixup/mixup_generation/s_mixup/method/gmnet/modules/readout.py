import torch
from torch import Tensor, nn
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from typing import Literal


class Readout(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        node_hiddens: list[int],
        graph_hiddens: list[int],
        use_gate: bool = True,
        pool_type: Literal["mean", "sum", "max"] = "sum",
    ):
        super().__init__()
        self.graph_feat_dim = node_hiddens[-1]
        self.use_gate = use_gate

        self.node_net = self._get_node_net(
            node_feat_dim, node_hiddens, use_gate
        )
        self.graph_net = self._get_graph_net(self.graph_feat_dim, graph_hiddens)

        if pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "sum":
            self.pool = global_add_pool
        elif pool_type == "max":
            self.pool = global_max_pool

    @staticmethod
    def _get_node_net(
        node_feat_dim: int, node_hiddens: list[int], use_gate: bool
    ) -> nn.Module:
        if use_gate:
            node_hiddens[-1] *= 2

        layers: list[nn.Module] = []
        layers.append(nn.Linear(node_feat_dim, node_hiddens[0]))
        for i in range(1, len(node_hiddens)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(node_hiddens[i - 1], node_hiddens[i]))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_graph_net(
        graph_feat_dim: int, graph_hiddens: list[int]
    ) -> nn.Module:
        layers: list[nn.Module] = []
        layers.append(nn.Linear(graph_feat_dim, graph_hiddens[0]))
        for i in range(1, len(graph_hiddens)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(graph_hiddens[i - 1], graph_hiddens[i]))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.node_net(x)

        if self.use_gate:
            gates = torch.sigmoid(
                x[:, : self.graph_feat_dim]
            )  # use first half for the gating
            x = (
                x[:, self.graph_feat_dim :] * gates
            )  # use second half for the node features

        graph_feat = self.pool(x, batch)
        graph_feat = self.graph_net(graph_feat)

        return graph_feat
