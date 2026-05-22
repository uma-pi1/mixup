from dataclasses import dataclass
from typing import Literal

from graph_mixup.augmentations.data.typing import DataModuleConfig
from graph_mixup.mixup_generation.typing import (
    MixupDatasetMethodConfig,
)
from graph_mixup.augmentations.typing import AugDatasetConfig

# ===
# GMNet
# ===

PoolType = Literal["mean", "sum", "max"]
FuseType = Literal["abs_diff", "add", "multiply", "concat", "cos"]
LossType = Literal["margin", "hamming"]


@dataclass
class LitGMNetConfig:
    num_layers: int


@dataclass
class LitGMNetTrainingConfig:
    batch_size: int


# ===
# S-Mixup Method
# ===

SimMethod = Literal["cos", "abs_diff"]
NormalizeMethod = Literal["softmax", "sinkhorn"]


@dataclass
class SMixupMethodConfig(MixupDatasetMethodConfig): ...


@dataclass
class SMixupDatasetConfig(AugDatasetConfig):
    lit_gmnet_config: LitGMNetConfig
    lit_gmnet_training_config: LitGMNetTrainingConfig


@dataclass
class SMixupDataModuleConfig(DataModuleConfig): ...
