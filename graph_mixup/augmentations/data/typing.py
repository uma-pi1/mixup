from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar, final

from torch_geometric.data import Data

from graph_mixup.config.typing import DatasetName

# ===
# Dataset Method Configs
# ===


@dataclass
class AbstractDatasetMethodConfig(ABC):
    """Base class of all method configs including the Nop-method."""


MethodConfigType = TypeVar("MethodConfigType")


@dataclass
@final
class NopDatasetMethodConfig(AbstractDatasetMethodConfig):
    """Config for a method without any parameters (e.g., vanilla)."""


# ===
# Dataset Config
# ===


@dataclass
class DatasetConfig(Generic[MethodConfigType]):
    """Base class of all dataset configs. Should not be used for mixup datasets."""

    method_config: MethodConfigType


DatasetConfigType = TypeVar("DatasetConfigType", bound=DatasetConfig)


@dataclass
class DataModuleConfig(ABC, Generic[DatasetConfigType]):
    dataset_config: DatasetConfigType
    data_dir: str
    dataset_name: DatasetName
    num_outer_folds: int
    num_inner_folds: int
    fold: int
    random_state: int
    num_workers: int
    batch_size: int
    eval_mode: bool
    device: int
    label_corruption_prob: float


DataModuleConfigType = TypeVar("DataModuleConfigType", bound=DataModuleConfig)


@dataclass
class SMixupItem:
    graph: Data
    lam: float
    source_indices: tuple[int, int]
    creation_time_us: int
