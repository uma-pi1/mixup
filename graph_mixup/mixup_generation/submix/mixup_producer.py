import time
import logging

import math
import random
from random import randint
from typing import Generator

import numpy as np
from graph_exporter.typing import SubMixConfig, MixupItem

from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.graph_import_database_handler import (
    GraphImportDatabaseHandler,
)
from graph_mixup.ged_database.models import Graph
from graph_mixup.mixup_generation.mixup_producer import MixupProducer
from graph_mixup.mixup_generation.submix.utils import (
    make_subgraph,
    get_cc_sizes,
    mix_graphs,
    NoEdgesException,
)

logger = logging.getLogger(__name__)


class SubMixProducer(MixupProducer):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        dataset_name: DatasetName,
        max_items_per_pair: int,
        aug_size: float,
    ) -> None:
        super().__init__(
            batch_size,
            seed,
            dataset_name,
            max_items_per_pair,
        )
        self.aug_size = aug_size
        self.db_manager = GraphImportDatabaseHandler()

    def produce_generator(self) -> Generator[Graph, None, None]:
        vanilla_graphs = self.db_manager.get_vanilla_graphs(self.dataset_name)

        random.seed(0)

        while True:
            idx0 = randint(0, len(vanilla_graphs) - 1)
            idx1 = randint(0, len(vanilla_graphs) - 1)
            graph0 = vanilla_graphs[idx0]
            graph1 = vanilla_graphs[idx1]

            try:
                yield self._mixup(graph0, graph1)
            except NoEdgesException:
                logger.info(
                    "Generated mixup graphs contains no edges. Skipping ..."
                )
                continue

    def _mixup(self, g1: Graph, g2: Graph) -> Graph:
        idx1, idx2 = g1.index, g2.index
        g1 = g1.get_pyg_data()
        g2 = g2.get_pyg_data()

        start_time = time.perf_counter()

        # Sample roots for PPR.
        root1 = randint(0, g1.num_nodes - 1)
        root2 = randint(0, g2.num_nodes - 1)

        # Sample subgraphs with PPR.
        subgraph1 = make_subgraph(g1, root1, get_cc_sizes(g1))
        subgraph2 = make_subgraph(g2, root2, get_cc_sizes(g2))
        aug_size = np.random.uniform(high=self.aug_size)
        aug_size = math.ceil(aug_size * min(len(subgraph1), len(subgraph2)))
        subgraph1 = subgraph1[:aug_size]
        subgraph2 = subgraph2[:aug_size]

        # Compute mixup graph.
        out, ratio = mix_graphs(g1, subgraph1, g2, subgraph2, label_by="edges")

        # Add graph to database.
        mixup_item = MixupItem(
            graph_dict=out.to_dict(),
            lam=ratio,
            source_indices=(idx1, idx2),
            creation_time_us=int((time.perf_counter() - start_time) * 1e6),
        )
        return self.db_manager.create_mixup_graph(
            mixup_item,
            SubMixConfig(seed=self.seed, aug_size=self.aug_size),
            self.dataset_name,
            PreBatchMixupName.SUBMIX,
            False,
        )
