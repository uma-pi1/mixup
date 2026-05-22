import logging
import random
from random import sample

import networkx as nx
from matplotlib import pyplot as plt

from graph_mixup.augmentations.data.typing import AbstractDatasetMethodConfig
from graph_mixup.augmentations.typing import GEDFilterFlags
from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.handlers.mixup_graph_fetcher import (
    MixupGraphFetcher,
)
from graph_mixup.plot_graphs.parser import parse


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def plot_mixup_graphs(
    dataset_name: DatasetName,
    method_name: PreBatchMixupName,
    seed: int,
) -> None:
    vanilla_graphs = BaseHandler().get_vanilla_graphs(dataset_name)
    random.seed(seed)
    vanilla_graphs_sample = sample(vanilla_graphs, 100)

    db_manager = MixupGraphFetcher(
        dataset_name,
        method_name,
        method_config=AbstractDatasetMethodConfig(),  # type: ignore
        parent_graph_ids=[g.graph_id for g in vanilla_graphs_sample],
        ged_filter_flags=GEDFilterFlags(
            max_ged_value=None,
            only_same_class=False,
            only_different_class=False,
            only_first_absolute_quintile=False,
            only_last_absolute_quintile=False,
            only_first_relative_quintile=False,
            only_last_relative_quintile=False,
        ),
        fetch_parents=True,
    )

    mixup_graphs = db_manager.fetch_mixup_graphs()
    random.seed(seed)
    mixup_graphs_sample = sample(mixup_graphs, 10)

    # ===
    # Create plots and store to disk.
    # ===

    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    for row, mixup_graph in enumerate(mixup_graphs_sample):
        logger.info(f"Row (starting at 1): {row + 1}")
        parent_0 = mixup_graph.mixup_attrs.parent_0
        parent_1 = mixup_graph.mixup_attrs.parent_1

        nx_parent_0 = parent_0.get_nx_graph()
        logger.info(f"Parent 0: {str(nx_parent_0)}")
        nx.draw(
            nx_parent_0,
            pos=nx.spring_layout(nx_parent_0, seed=seed),
            ax=axes[row, 0],
        )
        axes[row, 0].set_title(f"Parent 0 ({parent_0.graph_id})")

        nx_mixup_graph = mixup_graph.get_nx_graph()
        logger.info(f"Mixup Graph: {str(nx_mixup_graph)}")
        logger.info(f"Mixup Graph: {str(mixup_graph.get_ged_library_format())}")
        nx.draw(
            nx_mixup_graph,
            pos=nx.spring_layout(nx_mixup_graph, seed=seed),
            ax=axes[row, 1],
        )
        axes[row, 1].set_title(
            f"λ={mixup_graph.mixup_attrs.mixup_lambda} ({mixup_graph.graph_id})"
        )  # type: ignore

        nx_parent_1 = parent_1.get_nx_graph()
        logger.info(f"Parent 1: {str(nx_parent_1)}")
        nx.draw(
            nx_parent_1,
            pos=nx.spring_layout(nx_parent_1, seed=seed),
            ax=axes[row, 2],
        )
        axes[row, 2].set_title(f"Parent 1 ({parent_1.graph_id})")

    plt.tight_layout()
    plt.savefig(
        f"tmp/mixup_graphs_{dataset_name}_{method_name}_seed-{seed}.pdf"
    )
    plt.close(fig)


def main() -> None:
    args = parse()
    plot_mixup_graphs(args.dataset_name, args.method_name, args.seed)


if __name__ == "__main__":
    main()
