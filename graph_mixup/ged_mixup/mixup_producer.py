import logging
import time
from typing import Generator, assert_never

import numpy as np
from torch import Tensor
import networkx as nx
from networkx.algorithms.components import is_connected

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.handlers.ged_mixup_database_handler import (
    GEDMixupDatabaseHandler,
)
from graph_mixup.ged_database.models import Graph
from graph_mixup.ged_mixup.path_generator import PathGenerator
from graph_mixup.ged_mixup.typing import (
    EdgeDeletionOp,
    EdgeInsertionOp,
    EditOp,
    InvalidEditOpException,
    NodeInsertionOp,
    NodeRelabelOp,
    TempEdge,
    TempNode,
)
from graph_mixup.mixup_generation.mixup_producer import MixupProducer

logger = logging.getLogger(__name__)


def _compute_mixup_label(label0: list, label1: list, lam: float) -> list[float]:
    # list -> Tensor -> list
    label0 = Tensor(label0)
    label1 = Tensor(label1)
    mixup_label = (1 - lam) * label0 + lam * label1
    return mixup_label.tolist()


class GEDMixupProducer(MixupProducer):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        dataset_name: DatasetName,
        lam: float | None,
        mixup_alpha: float | None,
        max_items_per_pair: int,
        max_fail_count: int,
    ) -> None:
        super().__init__(
            batch_size,
            seed,
            dataset_name,
            max_items_per_pair,
        )
        self.lam = lam
        self.mixup_alpha = mixup_alpha
        self.max_fail_count = max_fail_count
        self.db_manager = GEDMixupDatabaseHandler()
        # Timing.
        self.path_creation_time_us: int | None = None

    def produce_generator(self) -> Generator[Graph, None, None]:
        for batch in self.db_manager.get_graph_pairs_with_ged(
            self.dataset_name,
            batch_size=self.batch_size,
            exclude_mixup=True,
        ):
            np.random.seed(self.seed)

            for g0, g1, ged in batch:
                path_generator = PathGenerator(
                    g0,
                    g1,
                    ged.get_mappings_dict(),
                    self.seed,
                )

                success_count, fail_count = 0, 0
                while True:
                    try:
                        # Check stopping criteria.
                        if success_count == self.max_items_per_pair:
                            break
                        if fail_count > self.max_fail_count:
                            logger.info(
                                f"\nMore than {self.max_fail_count} (ged={ged.value}). Advancing to next graph pair.\n"
                            )
                            break

                        # Create path and measure time.
                        path_start_time = time.perf_counter()
                        path = next(
                            path_generator.generate()
                        )  # <- path comes from here.
                        self.path_creation_time_us = int(
                            (time.perf_counter() - path_start_time) * 1e6
                        )

                        # Create mixup graph.
                        graph = self._generate_mixup_graph(g0, g1, path)

                        if graph is not None:
                            success_count += 1
                            yield graph
                        else:
                            fail_count += 1
                    except StopIteration:
                        break

    def _generate_mixup_graph(
        self, g0: Graph, g1: Graph, path: list[EditOp]
    ) -> Graph | None:
        assert g0.num_nodes() <= g1.num_nodes()

        # Start timing.
        generation_time_start = time.perf_counter()

        # ===
        # Step 1: Create a new graph by copying g0 (along with its nodes and
        #  edges).
        # ===
        nodes: dict[int, TempNode] = dict()
        for node in g0.nodes:
            nodes[node.node_id] = TempNode(node.node_id, node.attributes)

        edges: dict[tuple[int, int], TempEdge] = dict()
        for edge in g0.edges:
            edges[(edge.node_0_id, edge.node_1_id)] = TempEdge(
                edge.node_0_id, edge.node_1_id, edge.attributes
            )

        # ===
        # Step 2: Execute each edit operation (making sure that the graph
        #  remains valid throughout).
        # ===
        if self.lam is None:
            assert self.mixup_alpha is not None
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            assert self.mixup_alpha is None
            lam = self.lam

        logging.info(f"Mixup lambda: {lam}")

        mixup_path_len = round(len(path) * lam)
        for i, op in enumerate(path):
            if i == mixup_path_len:
                break

            if isinstance(op, NodeInsertionOp):
                # Does not affect validity of graph.
                nodes[op.image_node_id] = TempNode(
                    op.image_node_id,
                    list(op.new_attributes)
                    if op.new_attributes is not None
                    else None,
                )
            elif isinstance(op, EdgeInsertionOp):
                # Affects validity of graph if the nodes do not exist.
                if len(op.required_image_node_ids) > 0:
                    for node_id in op.required_image_node_ids:
                        if node_id not in nodes:
                            raise InvalidEditOpException(
                                f"EdgeInsertionOp {op} is invalid."
                            )

                edges[(op.id0, op.id1)] = TempEdge(
                    op.id0,
                    op.id1,
                    (
                        list(op.new_attributes)
                        if op.new_attributes is not None
                        else None
                    ),
                )

            elif isinstance(op, NodeRelabelOp):
                # Affects validity of graph if the node does not exist (would
                # result in KeyError).
                nodes[op.preimage_node_id] = TempNode(
                    op.preimage_node_id,
                    list(op.new_attributes)
                    if op.new_attributes is not None
                    else None,
                )
            elif isinstance(op, EdgeDeletionOp):
                # Does not affect validity of graph.
                del edges[(op.preimage_node_0_id, op.preimage_node_1_id)]
            else:
                assert_never(op)

        # ===
        # Step 3: Check connectivity.
        # ===
        nx_graph = nx.Graph()
        for nid in nodes.keys():
            nx_graph.add_node(nid)
        for u, v in edges.keys():
            nx_graph.add_edge(u, v)

        if not is_connected(nx_graph):
            logger.info(
                f"Generated mixup graph is not connected (nodes={len(nodes)}, edges={len(edges)}). Skipping insert."
            )
            return None

        # Stop timing.
        graph_generation_time_us = int(
            (time.perf_counter() - generation_time_start) * 1e6
        )

        # ===
        # Step 4: Store graph (along with its nodes, edges, and GED) in the
        #  database, and return the graph.
        # ===
        assert self.path_creation_time_us is not None
        graph_sql = self.db_manager.create_ged_mixup_graph(
            g0.dataset_id,
            g0.graph_id,
            g1.graph_id,
            lam,
            mixup_path_len,
            len(path),
            nodes,
            edges,
            _compute_mixup_label(g0.label, g1.label, lam),
            dict(
                mixup_alpha=self.mixup_alpha,
                max_items_per_pair=self.max_items_per_pair,
            ),
            creation_time_us=self.path_creation_time_us
            + graph_generation_time_us,
        )

        return graph_sql
