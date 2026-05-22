from typing import Literal
import torch
from torch import nn, Tensor
from torch_geometric.data import Batch

from .encoder import Encoder


class GraphMatchingNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        num_layers: int,
        fuse_type: Literal[
            "abs_diff", "add", "multiply", "concat", "cos"
        ] = "abs_diff",
        pool_type: Literal["mean", "sum", "max"] = "sum",
        hidden: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            node_feat_dim, num_layers, hidden, pool_type=pool_type, **kwargs
        )

        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.pred_head = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, 1),
                nn.Sigmoid(),
            )
        elif fuse_type == "cos":
            self.pred_head = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden),
            )
            self.cos = torch.nn.CosineSimilarity()
        else:
            in_hidden = hidden
            self.pred_head = nn.Sequential(
                nn.Linear(in_hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, data1: Batch, data2: Batch, pred_head=True
    ) -> Tensor | tuple[Tensor, Tensor]:
        embed1, embed2 = self.encoder(data1, data2, readout=True)

        if pred_head:
            if self.fuse_type == "add":
                pair_embed = embed1 + embed2
            elif self.fuse_type == "multiply":
                pair_embed = embed1 * embed2
            elif self.fuse_type == "concat":
                pair_embed = torch.cat((embed1, embed2), dim=1)
            elif self.fuse_type == "abs_diff":
                pair_embed = torch.abs(embed1 - embed2)
            elif self.fuse_type == "cos":
                embed1, embed2 = self.pred_head(embed1), self.pred_head(embed2)
                out = (1.0 + self.cos(embed1, embed2)) / 2.0
                return out.unsqueeze(1)

            return self.pred_head(pair_embed)

        else:
            return embed1, embed2
