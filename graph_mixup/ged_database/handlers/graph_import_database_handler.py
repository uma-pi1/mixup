import logging
from dataclasses import asdict
from math import isnan
from random import random

import torch
from graph_exporter.typing import BaseConfig, MixupItem
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch import Tensor
from torch_geometric.data.data import BaseData, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import one_hot
from tqdm import tqdm

from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.base_handler import (
    BaseHandler,
    DatasetNotFound,
)
from graph_mixup.ged_database.models import (
    Dataset,
    Edge,
    Graph,
    MixupAttr,
    Node,
)

logger = logging.getLogger(__name__)


class GraphNotFound(Exception): ...


class GraphImportDatabaseHandler(BaseHandler):
    @staticmethod
    def _get_graph(session: Session, dataset_id: int, idx: int) -> Graph:
        stmt = (
            select(Graph)
            .where(Graph.dataset_id == dataset_id)
            .where(Graph.index == idx)
        )

        res = session.scalar(stmt)

        if res is None:
            raise GraphNotFound()

        return res

    def get_or_create_dataset(
        self, dataset_name: DatasetName, data_dir: str
    ) -> Dataset:
        try:
            with Session(self._engine) as session:
                # ===
                # Check whether dataset already present in DB
                # ===

                dataset = self._get_dataset(session, dataset_name)
                print("Dataset already present in DB.")

        except DatasetNotFound:
            with Session(self._engine) as session, session.begin():
                print("Creating dataset in DB ...")

                # ===
                # Add dataset and all of its graphs to DB
                # ===

                pyg_dataset = TUDataset(
                    data_dir,
                    dataset_name,
                    use_node_attr=True,
                )

                num_classes = pyg_dataset.num_classes
                num_features = pyg_dataset.num_features
                dataset = Dataset(
                    name=dataset_name,
                    num_features=num_features if num_features > 0 else 1,
                    num_classes=num_classes,
                )
                session.add(dataset)
                session.flush()

                self._create_graphs(
                    session,
                    pyg_dataset,
                    dataset.dataset_id,
                    num_classes,
                    None,
                )

        return dataset

    @classmethod
    def _create_graphs(
        cls,
        session: Session,
        base_dataset: TUDataset,
        dataset_id: int,
        num_classes: int,
        sample_edges: bool,
    ) -> None:
        for idx, graph in tqdm(
            enumerate(base_dataset), total=len(base_dataset)
        ):
            cls._create_graph(
                session, graph, idx, dataset_id, num_classes, sample_edges
            )

    def create_mixup_graphs(
        self,
        mixup_items: list[MixupItem],
        config: BaseConfig,
        dataset_name: DatasetName,
        method_name: PreBatchMixupName,
        sample_edges: bool,
    ) -> list[int]:
        graph_ids: list[int] = []
        with Session(self._engine) as session, session.begin():
            for item in tqdm(mixup_items, leave=False):
                db_graph = self._create_mixup_graph(
                    session,
                    item,
                    config,
                    dataset_name,
                    method_name,
                    sample_edges,
                )
                graph_ids.append(db_graph.graph_id)
        return graph_ids

    def create_mixup_graph(
        self,
        item: MixupItem,
        config: BaseConfig,
        dataset_name: DatasetName,
        method_name: PreBatchMixupName,
        sample_edges: bool,
    ) -> Graph:
        with Session(self._engine) as session, session.begin():
            return self._create_mixup_graph(
                session,
                item,
                config,
                dataset_name,
                method_name,
                sample_edges,
            )

    @classmethod
    def _create_mixup_graph(
        cls,
        session: Session,
        item: MixupItem,
        config: BaseConfig,
        dataset_name: DatasetName,
        method_name: PreBatchMixupName,
        sample_edges: bool,
    ) -> Graph:
        dataset = cls._get_dataset(session, dataset_name)

        graph_sql = cls._create_graph(
            session,
            Data.from_dict(item.graph_dict),
            None,
            dataset.dataset_id,
            dataset.num_classes,
            sample_edges,
            one_hot_encode_label=False,
        )
        session.flush()

        # noinspection PyTypeChecker
        hparams = asdict(config)

        # Get parent graphs' IDs using their indices.
        p0 = cls._get_graph(session, dataset.dataset_id, item.source_indices[0])
        p1 = cls._get_graph(session, dataset.dataset_id, item.source_indices[1])

        session.add(
            MixupAttr.create(
                graph_sql.graph_id,
                item,
                p0.graph_id,
                p1.graph_id,
                method_name if not sample_edges else f"{method_name}_se",
                hparams,
                item.creation_time_us,
            )
        )

        return graph_sql

    @staticmethod
    def _create_graph(
        session: Session,
        graph: BaseData,
        idx: int | None,
        dataset_id: int,
        num_classes: int,
        sample_edges: bool,
        *,
        one_hot_encode_label: bool = True,
    ) -> Graph:
        # ===
        # Add graph
        # ===
        graph_sql = Graph(
            dataset_id=dataset_id,
            index=idx,
            label=(
                one_hot(graph.y, num_classes).tolist()
                if one_hot_encode_label
                else graph.y.tolist()
            ),
        )
        session.add(graph_sql)
        session.flush()

        # ===
        # Add nodes
        # ===

        nodes: list[Node] = []
        if graph.x is None:
            for node_idx in torch.unique(graph.edge_index):
                nodes.append(
                    Node(
                        graph_id=graph_sql.graph_id,
                        index=node_idx.item(),
                        attributes=torch.ones(1).tolist(),
                    )
                )
        else:
            node_feat: Tensor
            for idx_node, node_feat in enumerate(graph.x):
                attributes = [
                    None if isnan(x) else x for x in node_feat.tolist()
                ]
                nodes.append(
                    Node(
                        graph_id=graph_sql.graph_id,
                        index=idx_node,
                        attributes=attributes,
                    )
                )

        session.add_all(nodes)
        session.flush()

        # Create hash map to find the IDs of the edge's nodes given the
        # nodes' index.
        node_id_lookup: dict[int, int] = {}
        for node in nodes:
            node_id_lookup[node.index] = node.node_id

        # ===
        # Add edges
        # ===
        edges: list[Edge] = []
        # edge_index.t() has shape [num_edges, 2]
        edge: Tensor
        for idx_edge, edge in enumerate(graph.edge_index.t()):
            index0 = edge[0].item()
            index1 = edge[1].item()

            # Every edge appears in both directions, so adding one is
            # sufficient.
            if index0 < index1:
                if graph.edge_weight is None:
                    if sample_edges:
                        raise Exception(
                            "edges cannot be sampled without edge weights"
                        )

                    edges.append(
                        Edge(
                            graph_id=graph_sql.graph_id,
                            index=idx_edge,
                            node_0_id=node_id_lookup[index0],
                            node_1_id=node_id_lookup[index1],
                        )
                    )
                else:
                    weight = graph.edge_weight[idx_edge].item()

                    if sample_edges:
                        logger.debug(
                            f"Sampling edge with probability {weight} ..."
                        )

                        if weight < 0 or 1 < weight:
                            logger.debug("weight not in [0, 1]")

                        if random() < weight:
                            logger.debug("Successfully sampled.")
                            # ===
                            # Add edge with probability p='weight' (but without
                            # weight).
                            # ===
                            edges.append(
                                Edge(
                                    graph_id=graph_sql.graph_id,
                                    index=idx_edge,
                                    node_0_id=node_id_lookup[index0],
                                    node_1_id=node_id_lookup[index1],
                                )
                            )
                    else:
                        # ===
                        # Add edge with its edge weight.
                        # ===
                        edges.append(
                            Edge(
                                graph_id=graph_sql.graph_id,
                                index=idx_edge,
                                node_0_id=node_id_lookup[index0],
                                node_1_id=node_id_lookup[index1],
                                weight=weight,
                            )
                        )

        session.add_all(edges)

        return graph_sql
