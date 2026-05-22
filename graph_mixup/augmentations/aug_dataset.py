from abc import ABC, abstractmethod
from random import sample, choice
from typing import Generic

from torch_geometric.data import InMemoryDataset, Dataset, Data

from graph_mixup.augmentations.typing import AugDatasetConfigType


class AbstractAugDataset(ABC, InMemoryDataset, Generic[AugDatasetConfigType]):
    def __init__(self, dataset: Dataset, config: AugDatasetConfigType):
        super().__init__()
        self.dataset = dataset
        self.vanilla_indices = range(len(dataset))
        self.method_config = config.method_config

        self.num_vanilla = len(dataset) if config.use_vanilla else 0
        self.num_augmented = int(config.augmented_ratio * len(dataset))

        assert (
            config.augmented_ratio >= 0
        ), "augmentation ratio must be non-negative"
        assert self.is_graph_label_one_hot_encoded(
            dataset[0]
        ), "graph labels must be one-hot encoded"

    def get(self, idx: int) -> Data:
        if idx < self.num_vanilla:
            return self._transform_vanilla_item(self.dataset[idx])
        else:
            return self._get_aug_item(idx - self.num_vanilla)

    def len(self) -> int:
        return self.num_vanilla + self.num_augmented

    @staticmethod
    def is_graph_label_one_hot_encoded(item: Data) -> bool:
        return item.y.dim() > 1

    def _sample_vanilla_item(self) -> Data:
        return choice(self.dataset)

    def _sample_vanilla_pair(self) -> tuple[Data, Data]:
        idx0, idx1 = sample(self.vanilla_indices, 2)
        return self.dataset[idx0], self.dataset[idx1]

    def _transform_vanilla_item(self, item: Data) -> Data:
        """Allows to define special handling for vanilla dataset items to align with
        synthetic items (for instance adding edge_weights with value 1)."""
        item.num_nodes = item.x.size(0)
        item.edge_attr = None
        return item

    @abstractmethod
    def _get_aug_item(self, idx: int) -> Data:
        """Returns the idx-th synthetic item (idx ranges from 0 to self.num_augmented)."""
