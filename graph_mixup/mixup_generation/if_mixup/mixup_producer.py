import time
import random
from random import randint
from typing import Generator

import torch
from graph_exporter.typing import MixupItem, IfMixupConfig
from torch.distributions import Beta
from torch_geometric.data import Data

from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.graph_import_database_handler import (
    GraphImportDatabaseHandler,
)
from graph_mixup.ged_database.models import Graph
from graph_mixup.mixup_generation.mixup_producer import MixupProducer


class IfMixupProducer(MixupProducer):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        dataset_name: DatasetName,
        lam: float | None,
        mixup_alpha: float | None,
        max_items_per_pair: int,
        sample_edges: bool,
    ) -> None:
        super().__init__(
            batch_size,
            seed,
            dataset_name,
            max_items_per_pair,
        )
        self.lam = lam
        self.mixup_alpha = mixup_alpha
        self.sample_edges = sample_edges
        self.db_manager = GraphImportDatabaseHandler()

    def produce_generator(self) -> Generator[Graph, None, None]:
        vanilla_graphs = self.db_manager.get_vanilla_graphs(self.dataset_name)

        random.seed(0)

        while True:
            idx0 = randint(0, len(vanilla_graphs) - 1)
            idx1 = randint(0, len(vanilla_graphs) - 1)
            graph0 = vanilla_graphs[idx0]
            graph1 = vanilla_graphs[idx1]

            yield self._mixup(graph0, graph1)

    def _mixup(self, g1: Graph, g2: Graph) -> Graph:
        idx1, idx2 = g1.index, g2.index
        g1 = g1.get_pyg_data()
        g2 = g2.get_pyg_data()

        start_time = time.perf_counter()

        # Assume: Graph labels are one-hot encoded
        assert g1.y.dim() > 1, "graph labels must be one-hot encoded"

        # Assume: Input graphs have no edge weights
        assert (
            g1.edge_weight is None
        ), "input graphs with edge weights are currently unsupported"

        if g1.num_nodes < g2.num_nodes:
            g_small, g_large = g1, g2
        else:
            g_small, g_large = g2, g1
            idx1, idx2 = idx2, idx1

        # Get sparse matrices of same size
        adj_small = torch.sparse_coo_tensor(
            indices=g_small.edge_index,
            values=torch.ones(g_small.edge_index.size(1)),
            size=(g_large.num_nodes, g_large.num_nodes),
        )
        adj_large = torch.sparse_coo_tensor(
            indices=g_large.edge_index,
            values=torch.ones(g_large.edge_index.size(1)),
            size=(g_large.num_nodes, g_large.num_nodes),
        )

        # Sample mixup parameter
        lam = self.lam
        if lam is None:
            lam = Beta(self.mixup_alpha, self.mixup_alpha).sample()
        else:
            lam = torch.tensor(lam)

        # Compute convex combinations
        adj_matrix = (lam * adj_small + (1 - lam) * adj_large).coalesce()
        y = lam * g_small.y + (1 - lam) * g_large.y

        # Create new Data object
        mixup_graph = Data(
            edge_index=adj_matrix.indices(),
            edge_weight=adj_matrix.values(),
            y=y,
        )

        attr_keys = g_small.keys()
        num_nodes_diff = g_large.num_nodes - g_small.num_nodes

        # Add additional parameters
        if "x" in attr_keys:
            x_small_padded = torch.cat(
                (g_small.x, torch.zeros(num_nodes_diff, g_small.num_features))
            )
            x = lam * x_small_padded + (1 - lam) * g_large.x
            mixup_graph.x = x
            mixup_graph.num_nodes = x.size(0)

        handled_attr = {
            "edge_index",
            "edge_weight",
            "x",
            "y",
            "num_nodes",
            "edge_attr",
        }  # ignore edge_attr
        unhandled_attr = set(attr_keys).difference(handled_attr)

        # Assume: only attributes in handled_attr are supported
        assert (
            len(unhandled_attr) == 0
        ), f"unhandled graph attributes: {unhandled_attr}"

        mixup_item = MixupItem(
            graph_dict=mixup_graph.to_dict(),
            lam=lam.item(),
            source_indices=(idx1, idx2),
            creation_time_us=int((time.perf_counter() - start_time) * 1e6),
        )

        return self.db_manager.create_mixup_graph(
            mixup_item,
            IfMixupConfig(seed=self.seed, mixup_alpha=self.mixup_alpha),
            self.dataset_name,
            PreBatchMixupName.IF_MIXUP,
            self.sample_edges,
        )
