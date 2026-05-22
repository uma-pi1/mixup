from torch_geometric.data import Dataset
from typing_extensions import override

from graph_mixup.augmentations.basic.dataset import BasicAugDataset
from graph_mixup.augmentations.data.abstract_aug_data_module import (
    AbstractAugDataModule,
)


class BasicAugDataModule(AbstractAugDataModule):
    @override
    def transform_train_set(self, train_set: Dataset) -> Dataset:
        return BasicAugDataset(train_set, self.config.dataset_config)
