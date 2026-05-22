import os
from torch_geometric.datasets import TUDataset
import json

from graph_mixup.augmentations.data.utils import compute_max_degree_mean_std

datasets = [
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
    "PROTEINS",
    "MUTAG",
    "Mutagenicity",
    "NCI1",
    "DD",
    "ENZYMES",
    "COLLAB",
]

statistics = {}

for dataset_name in datasets:
    dataset = TUDataset("data", dataset_name, use_node_attr=True)

    max_degree, mean, std = compute_max_degree_mean_std(dataset)

    statistics[dataset_name] = {
        "max_degree": max_degree,
        "mean": mean,
        "std": std,
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "num_graphs": len(dataset),
    }

path = os.path.join(
    os.path.dirname(__file__),
    "dataset_statistics.json",
)
json.dump(statistics, open(path, "w"), indent=4)
