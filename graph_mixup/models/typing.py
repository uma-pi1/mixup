from abc import ABC
from dataclasses import dataclass

from graph_mixup.typing import BaseEnum


class OptimizerName(BaseEnum):
    ADAM = "ADAM"
    SGD = "SGD"


class AggrName(BaseEnum):
    ADD = "ADD"
    MEAN = "MEAN"
    MAX = "MAX"


class NormName(BaseEnum):
    BATCH_NORM = "BATCH_NORM"


@dataclass
class AbstractModelMethodConfig(ABC): ...


@dataclass
class ModelMixupConfig(AbstractModelMethodConfig):
    mixup_alpha: float
    use_vanilla: bool
    augmented_ratio: float


@dataclass
class NopModelMethodConfig(AbstractModelMethodConfig): ...


@dataclass
class GNNParams:
    optimizer: OptimizerName
    lr: float
    num_conv_layers: int
    hidden_channels: int
    aggr: AggrName
    dropout: float
    norm: NormName | None
    num_pre_processing_layers: int
    num_post_processing_layers: int
    method_config: AbstractModelMethodConfig
    seed: int


@dataclass
class BaselineParams(GNNParams):
    batch_size: int

    def get_gnn_params(self) -> GNNParams:
        params = vars(self).copy()
        del params["batch_size"]
        return GNNParams(**params)
