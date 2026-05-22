from abc import ABC
from typing import Union

from lightning import seed_everything
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP, GCN, GIN
from torch_geometric.nn.pool import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from typing_extensions import assert_never

from graph_mixup.mixup_generation import emb_mixup
from graph_mixup.mixup_generation.emb_mixup.typing import (
    EmbMixupModelConfig,
)
from graph_mixup.models.typing import AggrName, GNNParams, NopModelMethodConfig


def resolve_aggr_fn(
    aggr: AggrName,
) -> Union[global_add_pool, global_mean_pool, global_max_pool]:
    if aggr is AggrName.ADD:
        return global_add_pool
    elif aggr is AggrName.MEAN:
        return global_mean_pool
    elif aggr is AggrName.MAX:
        return global_max_pool
    else:
        assert_never(aggr)


class PygGNN(ABC, Module):
    def __init__(
        self,
        in_channels: int,
        encoder: Module,
        out_channels: int,
        params: GNNParams,
    ):
        super().__init__()
        self.method_config = params.method_config

        self.pre_processing_layers = MLP(
            in_channels=in_channels,
            hidden_channels=params.hidden_channels,
            out_channels=params.hidden_channels,
            num_layers=params.num_pre_processing_layers,
            dropout=params.dropout,
            norm=params.norm,
        )

        self.encoder = encoder
        self.readout = resolve_aggr_fn(params.aggr)

        self.post_processing_layers = MLP(
            in_channels=params.hidden_channels,
            hidden_channels=params.hidden_channels,
            out_channels=out_channels,
            num_layers=params.num_post_processing_layers,
            dropout=params.dropout,
            norm=params.norm,
        )

    def after_readout(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """While in training and a model mixup method is given, apply the
        mixup method."""
        if self.training and not isinstance(
            self.method_config, NopModelMethodConfig
        ):
            if isinstance(self.method_config, EmbMixupModelConfig):
                return emb_mixup.mixup(x, y, self.method_config)

            raise TypeError("mixup model config not implemented / unknown")

        return x, y

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        out = self.pre_processing_layers(batch.x)

        if self.encoder.supports_edge_weight:
            out = self.encoder(out, batch.edge_index, batch.edge_weight)
        else:
            out = self.encoder(out, batch.edge_index)

        out = self.readout(out, batch.batch)
        out, labels = self.after_readout(out, batch.y)
        out = self.post_processing_layers(out)
        return out, labels


class GCNClassification(PygGNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        params: GNNParams,
        test_round: int | None,
    ):
        seed_everything(
            params.seed if test_round is None else params.seed + test_round
        )

        encoder = GCN(
            params.hidden_channels,
            params.hidden_channels,
            params.num_conv_layers,
            params.hidden_channels,
            params.dropout,
            norm=params.norm,
        )
        super().__init__(in_channels, encoder, out_channels, params)


class GINClassification(PygGNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        params: GNNParams,
        test_round: int | None,
    ):
        seed_everything(
            params.seed if test_round is None else params.seed + test_round
        )

        encoder = GIN(
            params.hidden_channels,
            params.hidden_channels,
            params.num_conv_layers,
            params.hidden_channels,
            params.dropout,
            norm=params.norm,
        )
        super().__init__(in_channels, encoder, out_channels, params)
