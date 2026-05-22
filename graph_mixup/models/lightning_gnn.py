from dataclasses import asdict
from typing import Literal, assert_never

import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer, Adam, SGD
from torch_geometric.data import Batch

from graph_mixup.config.typing import ModelName
from graph_mixup.models.pyg_gnn import (
    GCNClassification,
    GINClassification,
    PygGNN,
)
from graph_mixup.models.typing import GNNParams, OptimizerName, NormName


class LitGNN(LightningModule):
    def __init__(
        self,
        model: ModelName,
        in_channels: int,
        out_channels: int,
        gnn_params: GNNParams,
        test_round: int | None,
    ):
        super().__init__()
        assert in_channels > 0 and out_channels > 0
        self.save_hyperparameters(dict(model_config=asdict(gnn_params)))

        self.optimizer_name = gnn_params.optimizer
        self.lr = gnn_params.lr

        self.uses_batch_norm = gnn_params.norm is NormName.BATCH_NORM
        self.method_config = gnn_params.method_config

        if model == "GCN":
            self.gnn: PygGNN = GCNClassification(
                in_channels, out_channels, gnn_params, test_round
            )
        elif model == "GIN":
            self.gnn = GINClassification(
                in_channels, out_channels, gnn_params, test_round
            )
        else:
            raise ValueError(f"Invalid model: {model}")

    def configure_optimizers(self) -> Optimizer:
        if self.optimizer_name is OptimizerName.ADAM:
            return Adam(self.gnn.parameters(), lr=self.lr)

        if self.optimizer_name is OptimizerName.SGD:
            return SGD(self.gnn.parameters(), lr=self.lr)

        assert_never(self.optimizer_name)

    def training_step(self, batch: Batch) -> Tensor:
        out, labels = self.gnn(batch)
        loss = F.cross_entropy(out, labels)
        self.log("train_loss", loss, batch_size=batch.batch_size, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch) -> None:
        return self._eval(batch, "val")

    def test_step(self, batch: Batch) -> None:
        return self._eval(batch, "test")

    def _eval(self, batch, mode: Literal["val", "test"]) -> None:
        # Compute loss and accuracy
        out, _ = self.gnn(batch)
        loss = F.cross_entropy(out, batch.y)
        acc = out.argmax(dim=-1).eq(batch.y.argmax(dim=-1)).float().mean()

        # Log metrics
        self.log(f"{mode}_loss", loss, batch_size=batch.batch_size)
        self.log(f"{mode}_acc", acc, prog_bar=True, batch_size=batch.batch_size)
        if mode == "val":
            self.log("hp_metric", acc, batch_size=batch.batch_size)
