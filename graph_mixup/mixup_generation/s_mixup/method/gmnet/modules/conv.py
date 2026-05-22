from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_scatter import scatter


class GMNConv(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        message_net_hiddens: list[int],
        update_net_hiddens: list[int],
        node_update_type: Literal["mlp", "residual", "gru"] = "residual",
        layer_norm=False,
    ):
        super().__init__()

        self.node_update_type = node_update_type

        self.message_net = self._get_message_net(
            node_feat_dim, message_net_hiddens
        )
        self.update_net = self._get_update_net(
            node_feat_dim, update_net_hiddens, node_update_type
        )
        self.message_norm = (
            nn.LayerNorm(message_net_hiddens[-1])
            if layer_norm
            else nn.Identity()
        )
        self.update_norm = (
            nn.LayerNorm(node_feat_dim) if layer_norm else nn.Identity()
        )

    @staticmethod
    def _get_message_net(
        node_feat_dim: int, message_net_hiddens: list[int]
    ) -> nn.Module:
        layer: list[nn.Module] = []
        layer.append(nn.Linear(node_feat_dim * 2, message_net_hiddens[0]))
        for i in range(1, len(message_net_hiddens)):
            layer.append(nn.ReLU())
            layer.append(
                nn.Linear(message_net_hiddens[i - 1], message_net_hiddens[i])
            )
        return nn.Sequential(*layer)

    @staticmethod
    def _get_update_net(
        node_feat_dim: int,
        update_net_hiddens: list[int],
        node_update_type: Literal["mlp", "residual", "gru"],
    ) -> nn.Module:
        if node_update_type == "gru":
            return nn.GRU(node_feat_dim * 3, node_feat_dim)

        layer: list[nn.Module] = []
        layer.append(nn.Linear(node_feat_dim * 4, update_net_hiddens[0]))
        for i in range(1, len(update_net_hiddens)):
            layer.append(nn.ReLU())
            layer.append(
                nn.Linear(update_net_hiddens[i - 1], update_net_hiddens[i])
            )
        layer.append(nn.ReLU())
        layer.append(nn.Linear(update_net_hiddens[-1], node_feat_dim))
        return nn.Sequential(*layer)

    def message_aggr(self, x: Tensor, edge_index: Tensor) -> Tensor:
        target_idx, source_idx = edge_index
        message_inputs = torch.cat(
            (x.index_select(0, source_idx), x.index_select(0, target_idx)),
            dim=-1,
        )
        messages = self.message_net(message_inputs)
        aggregation = scatter(
            messages, target_idx, dim=0, dim_size=x.shape[0], reduce="add"
        )

        return self.message_norm(aggregation)

    def node_update(
        self, x: Tensor, messages: Tensor, attentions: Tensor
    ) -> Tensor:
        if self.node_update_type == "gru":
            update_inputs = torch.cat((messages, attentions), dim=-1)
            _, new_x = self.update_net(
                update_inputs.unsqueeze(0), x.unsqueeze(0)
            )
            new_x = torch.squeeze(new_x, dim=0)
        else:
            update_inputs = torch.cat((messages, attentions, x), dim=-1)
            new_x = self.update_net(update_inputs)

        new_x = self.update_norm(new_x)

        if self.node_update_type == "residual":
            return x + new_x
        return new_x

    def cross_attention(
        self, x1: Tensor, batch1: Tensor, x2: Tensor, batch2: Tensor
    ) -> tuple[Tensor, Tensor]:
        att_scores = torch.mm(x1, x2.T)
        att_scores[batch1.view(-1, 1) != batch2.view(1, -1)] = -float("inf")

        att_weights1 = F.softmax(att_scores, dim=1)
        att_weights2 = F.softmax(att_scores.T, dim=1)
        attentions1 = x1 - torch.mm(att_weights1, x2)
        attentions2 = x2 - torch.mm(att_weights2, x1)

        return attentions1, attentions2

    def forward(
        self,
        x1: Tensor,
        edge_index1: Tensor,
        batch1: Tensor,
        x2: Tensor,
        edge_index2: Tensor,
        batch2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        messages1 = self.message_aggr(x1, edge_index1)
        messages2 = self.message_aggr(x2, edge_index2)

        attentions1, attentions2 = self.cross_attention(x1, batch1, x2, batch2)

        new_x1 = self.node_update(x1, messages1, attentions1)
        new_x2 = self.node_update(x2, messages2, attentions2)

        return new_x1, new_x2
