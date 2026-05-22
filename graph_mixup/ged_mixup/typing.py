from dataclasses import dataclass
from typing import Union

from graph_mixup.mixup_generation.typing import (
    MixupDatasetMethodConfig,
)


@dataclass(frozen=True)
class NodeRelabelOp:
    preimage_node_id: int
    image_node_id: int
    new_attributes: tuple[float, ...] | None


@dataclass(frozen=True)
class EdgeDeletionOp:
    preimage_node_0_id: int
    preimage_node_1_id: int


@dataclass(frozen=True)
class NodeInsertionOp:
    image_node_id: int
    new_attributes: tuple[float, ...] | None


@dataclass(frozen=True)
class EdgeInsertionOp:
    id0: int
    id1: int
    new_attributes: tuple[float, ...] | None
    required_image_node_ids: frozenset[int]


EditOp = Union[NodeRelabelOp, EdgeDeletionOp, NodeInsertionOp, EdgeInsertionOp]


@dataclass(frozen=True)
class PathOps:
    node_relabel_ops: set[NodeRelabelOp]
    edge_deletion_ops: set[EdgeDeletionOp]
    node_insertion_ops: set[NodeInsertionOp]
    edge_insertion_ops: set[EdgeInsertionOp]

    def get_all(self) -> set[EditOp]:
        return (
            self.node_relabel_ops
            | self.edge_deletion_ops
            | self.node_insertion_ops
            | self.edge_insertion_ops
        )

    def __len__(self) -> int:
        return sum(len(op_set) for op_set in self.__dict__.values())


class InvalidEditOpException(Exception): ...


@dataclass
class TempNode:
    id: int
    attributes: list[float] | None


@dataclass
class TempEdge:
    id0: int
    id1: int
    attributes: list[float] | None


@dataclass
class GEDMixupMethodConfig(MixupDatasetMethodConfig):
    # TODO: Inherit from BaseConfig instead (requires seed to be stored during
    #  graph generation).
    mixup_alpha: float
    max_items_per_pair: int
