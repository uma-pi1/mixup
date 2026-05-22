from abc import ABC
from dataclasses import dataclass

from typing_extensions import final

from graph_mixup.augmentations.data.typing import AbstractDatasetMethodConfig


@dataclass
class BasicAugDatasetMethodConfig(AbstractDatasetMethodConfig, ABC): ...


@dataclass
@final
class DropEdgeDatasetMethodConfig(BasicAugDatasetMethodConfig):
    p: float


@dataclass
@final
class DropNodeDatasetMethodConfig(BasicAugDatasetMethodConfig):
    p: float


@dataclass
@final
class DropPathDatasetMethodConfig(BasicAugDatasetMethodConfig):
    p: float
    walks_per_node: int
    walk_length: int


@dataclass
@final
class PerturbNodeAttrDatasetMethodConfig(BasicAugDatasetMethodConfig):
    std: float
