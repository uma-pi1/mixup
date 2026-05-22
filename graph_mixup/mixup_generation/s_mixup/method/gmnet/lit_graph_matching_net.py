import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Batch

from graph_mixup.mixup_generation.s_mixup.method.gmnet.modules.graph_matching_net import (
    GraphMatchingNet,
)
from graph_mixup.mixup_generation.s_mixup.typing import (
    LitGMNetConfig,
    LossType,
)
from graph_mixup.mixup_generation.s_mixup.utils import triplet_loss


class LitGraphMatchingNet(LightningModule):
    def __init__(
        self,
        node_feat_dim: int,
        config: LitGMNetConfig,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        loss_type: LossType = "margin",
    ) -> None:
        super().__init__()
        self.model = GraphMatchingNet(
            node_feat_dim,
            config.num_layers,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(
        self, data1: Batch, data2: Batch, pred_head=True
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.model(data1, data2, pred_head)

    def training_step(
        self, batch: tuple[Batch, Batch, Batch], batch_idx
    ) -> Tensor:
        anchor_data, pos_data, neg_data = batch

        x_1, y = self.model(anchor_data, pos_data, pred_head=False)
        x_2, z = self.model(anchor_data, neg_data, pred_head=False)

        loss = triplet_loss(x_1, y, x_2, z, loss_type=self.loss_type)
        loss = torch.mean(loss)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=anchor_data.batch_size,
        )
        self.log("hp_metric", loss)
        return loss
