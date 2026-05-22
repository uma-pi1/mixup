import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from graph_mixup.augmentations.basic.typing import (
    PerturbNodeAttrDatasetMethodConfig,
)


class PerturbNodeAttr(BaseTransform):
    def __init__(self, params: PerturbNodeAttrDatasetMethodConfig) -> None:
        self.std = params.std

    def forward(self, data: Data) -> Data:
        data.x = data.x + torch.empty_like(data.x).normal_(
            mean=0.0, std=self.std
        )
        return data
