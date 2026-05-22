from dataclasses import dataclass
from typing import Optional, Any

import networkx as nx
import torch
from graph_exporter.typing import MixupItem
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    JSON,
    UniqueConstraint,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index


class Base(DeclarativeBase): ...


class Dataset(Base):
    __tablename__ = "datasets"

    # ===
    # Columns
    # ===

    dataset_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    num_features = Column(Integer, nullable=False)
    num_classes = Column(Integer, nullable=False)

    # ===
    # Relationships
    # ===

    graphs: Mapped[list["Graph"]] = relationship(back_populates="dataset")


class Graph(Base):
    __tablename__ = "graphs"
    __table_args__ = (UniqueConstraint("dataset_id", "index"),)

    # ===
    # Columns
    # ===

    graph_id = Column(Integer, primary_key=True)
    index = Column(Integer, nullable=True)
    dataset_id = Column(
        Integer, ForeignKey("datasets.dataset_id"), nullable=False
    )
    label = Column(JSON, nullable=False)

    # ===
    # Relationships
    # ===

    dataset: Mapped[Dataset] = relationship(
        foreign_keys=[dataset_id], back_populates="graphs"
    )
    mixup_attrs: Mapped[Optional["MixupAttr"]] = relationship(
        foreign_keys="MixupAttr.graph_id"
    )
    nodes: Mapped[list["Node"]] = relationship(order_by="Node.index")
    edges: Mapped[list["Edge"]] = relationship()

    # ===
    # Methods
    # ===

    def _get_node_features(self) -> Tensor | None:
        features = [node.attributes for node in self.nodes]

        if features[0] is None:
            return None

        return torch.tensor(features)

    def _get_unsorted_edge_index(self) -> Tensor:
        # ===
        # This intentionally includes both directions of each edge (as required
        # by PyG).
        # ===
        edges = [
            (edge.node_0.index, edge.node_1.index) for edge in self.edges
        ] + [(edge.node_1.index, edge.node_0.index) for edge in self.edges]
        return torch.tensor(edges).t()

    def _get_edge_index(self) -> Tensor:
        edge_index_unsorted = self._get_unsorted_edge_index()

        if len(edge_index_unsorted) == 0:
            return torch.tensor([[], []], dtype=torch.long)

        return sort_edge_index(edge_index_unsorted)

    def _get_edge_weight(self) -> Tensor | None:
        # First get edge_weights (once for each direction).
        unsorted_weights = [edge.weight for edge in self.edges] + [
            edge.weight for edge in self.edges
        ]

        if not unsorted_weights or unsorted_weights[0] is None:
            return None

        # Then get unsorted edge index.
        edge_index_unsorted = self._get_unsorted_edge_index()

        # Sort weights and edge index together.
        _, sorted_weights = sort_edge_index(
            edge_index_unsorted, torch.tensor(unsorted_weights)
        )

        return sorted_weights

    def __str__(self) -> str:
        return self.get_ged_library_format()

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def node_attributes_with_default_value(
        self,
    ) -> dict[int, tuple[float, ...]]:
        return {
            node.index: tuple(
                node.attributes if node.attributes is not None else [0.0]
            )
            for node in self.nodes
        }

    def get_pyg_data(self) -> Data:
        edge_weight = self._get_edge_weight()

        if edge_weight is None:
            return Data(
                x=self._get_node_features(),
                edge_index=self._get_edge_index(),
                y=torch.tensor(self.label),
            )

        return Data(
            x=self._get_node_features(),
            edge_index=self._get_edge_index(),
            edge_weight=edge_weight,
            y=torch.tensor(self.label),
        )

    def get_nx_graph(self) -> nx.Graph:
        g = nx.Graph()
        for node in self.nodes:
            g.add_node(
                node.index,
                x=node.attributes,
            )
        for edge in self.edges:
            if edge.weight is None:
                g.add_edge(
                    edge.node_0.index,
                    edge.node_1.index,
                    label=edge.attributes,
                )
            else:
                g.add_edge(
                    edge.node_0.index,
                    edge.node_1.index,
                    weight=edge.weight,
                    label=edge.attributes,
                )
        return g

    def get_ged_library_format(self) -> str:
        # Start the graph entry
        ged_format = [f"t {self.graph_id} {self.dataset_id}"]

        # ===
        # Add vertices. Make sure that the list of attributes contains no
        # whitespaces to ensure compatibility with the GED binary.
        # ===
        for node in self.nodes:
            node_attr = (
                str(node.attributes).replace(" ", "")
                if node.attributes is not None
                else 0
            )
            ged_format.append(f"v {node.index} {node_attr}")

        # ===
        # Add edges (whitespace needs to be handled due to the same reason
        # as for nodes' attributes above).
        # ===
        for edge in self.edges:
            if edge.weight is not None:
                edge_attr = edge.weight
            elif edge.attributes is not None:
                edge_attr = str(edge.attributes).replace(" ", "")
            else:
                edge_attr = 0

            ged_format.append(
                f"e {edge.node_0.index} {edge.node_1.index} {edge_attr}"
            )

        # Join all lines into a single string
        return "\n".join(ged_format)


class Node(Base):
    __tablename__ = "nodes"
    __table_args__ = (UniqueConstraint("graph_id", "index"),)

    # ===
    # Columns
    # ===

    node_id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, ForeignKey("graphs.graph_id"), nullable=False)
    index = Column(Integer, nullable=False)
    attributes = Column(JSON, nullable=True)


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint("graph_id", "index"),
        UniqueConstraint("graph_id", "node_0_id", "node_1_id"),
        CheckConstraint(
            "node_0_id < node_1_id", "node_0_id_less_than_node_1_id"
        ),
    )

    # ===
    # Columns
    # ===

    edge_id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, ForeignKey("graphs.graph_id"), nullable=False)
    node_0_id = Column(Integer, ForeignKey("nodes.node_id"), nullable=False)
    node_1_id = Column(Integer, ForeignKey("nodes.node_id"), nullable=False)
    index = Column(Integer, nullable=False)
    attributes = Column(JSON, nullable=True)
    weight = Column(Float, nullable=True)
    # ===
    # Relationships
    # ===
    node_0: Mapped[Node] = relationship(foreign_keys=[node_0_id])
    node_1: Mapped[Node] = relationship(foreign_keys=[node_1_id])


@dataclass
class DBMixupItem:
    graph: Data
    dataset_id: int
    p0: Graph
    p1: Graph
    lam: float
    method_name: str


class MixupAttr(Base):
    __tablename__ = "mixup_attrs"
    __table_args__ = (Index("ix_mixup_method", "mixup_method"),)

    # ===
    # Columns
    # ===

    mixup_attr_id = Column(Integer, primary_key=True)
    graph_id = Column(
        Integer, ForeignKey("graphs.graph_id"), nullable=False, unique=True
    )
    parent_0_id = Column(Integer, ForeignKey("graphs.graph_id"))
    parent_1_id = Column(Integer, ForeignKey("graphs.graph_id"))
    mixup_lambda = Column(Float, nullable=True)
    mixup_method = Column(String(255), nullable=False)
    mixup_hyperparameters = Column(JSON, nullable=True)
    creation_time_us = Column(Integer, nullable=True)

    # ===
    # Relationships
    # ===

    graph: Mapped[Graph] = relationship(
        foreign_keys=[graph_id], back_populates="mixup_attrs"
    )

    parent_0: Mapped[Graph] = relationship(foreign_keys=[parent_0_id])
    parent_1: Mapped[Graph] = relationship(foreign_keys=[parent_1_id])

    # ===
    # Methods
    # ===

    @classmethod
    def create(
        cls,
        graph_id: int,
        item: MixupItem,
        p0_id: int,
        p1_id: int,
        method_name: str,
        hparams: dict[str, Any],
        creation_time_us: int | None = None,
    ) -> "MixupAttr":
        return cls(
            graph_id=graph_id,
            parent_0_id=p0_id,
            parent_1_id=p1_id,
            mixup_lambda=item.lam,
            mixup_method=method_name,
            mixup_hyperparameters=hparams,
            creation_time_us=creation_time_us,
        )


class GED(Base):
    __tablename__ = "geds"
    __table_args__ = (UniqueConstraint("graph_0_id", "graph_1_id"),)
    ged_id = Column(Integer, primary_key=True)
    graph_0_id = Column(Integer, ForeignKey("graphs.graph_id"))
    graph_1_id = Column(Integer, ForeignKey("graphs.graph_id"))
    value = Column(Integer, nullable=False)
    time = Column(Integer, nullable=True)
    lower_bound = Column(Integer, nullable=True)

    # ===
    # Relationships
    # ===

    graph_0: Mapped[Graph] = relationship(foreign_keys=[graph_0_id])
    graph_1: Mapped[Graph] = relationship(foreign_keys=[graph_1_id])
    mappings: Mapped[list["Mapping"]] = relationship(
        foreign_keys="Mapping.ged_id", back_populates="ged"
    )

    # ===
    # Methods
    # ===

    def get_mappings_dict(self) -> dict[int, int]:
        return {
            mapping.node_0_id: mapping.node_1_id for mapping in self.mappings
        }


class Mapping(Base):
    __tablename__ = "mappings"
    __table_args__ = (
        UniqueConstraint("ged_id", "node_0_id"),
        UniqueConstraint("ged_id", "node_1_id"),
    )
    mapping_id = Column(Integer, primary_key=True)
    ged_id = Column(Integer, ForeignKey("geds.ged_id"), nullable=False)
    node_0_id = Column(Integer, ForeignKey("nodes.node_id"), nullable=False)
    node_1_id = Column(Integer, ForeignKey("nodes.node_id"), nullable=False)

    # ===
    # Relationships
    # ===

    ged: Mapped[GED] = relationship(
        foreign_keys="Mapping.ged_id",
        back_populates="mappings",
    )
    node_0: Mapped[Node] = relationship(foreign_keys="Mapping.node_0_id")
    node_1: Mapped[Node] = relationship(foreign_keys="Mapping.node_1_id")
