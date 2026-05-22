from typing import Literal
import torch
from torch import nn, Tensor
from torch_geometric.data import Batch

from .readout import Readout
from .conv import GMNConv


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        hidden: int,
        pool_type: Literal["mean", "sum", "max"] = "sum",
        use_gate=True,
        node_update_type: Literal["mlp", "residual", "gru"] = "residual",
        layer_norm=False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden),
        )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GMNConv(
                    hidden,
                    [hidden * 2, hidden * 2],
                    [hidden * 2],
                    node_update_type,
                    layer_norm,
                )
            )

        self.readout = Readout(hidden, [hidden], [hidden], use_gate, pool_type)

    def forward(
        self, data1: Batch, data2: Batch, readout=True
    ) -> tuple[Tensor, Tensor]:
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        x1, x2 = self.embedding(x1), self.embedding(x2)

        for conv in self.convs:
            x1, x2 = conv(x1, edge_index1, batch1, x2, edge_index2, batch2)

        if not readout:
            return x1, x2

        return self.readout(x1, batch1), self.readout(x2, batch2)
