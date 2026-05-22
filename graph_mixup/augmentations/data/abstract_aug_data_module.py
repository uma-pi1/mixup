import json
import os.path as osp
from abc import abstractmethod
from dataclasses import asdict
from enum import Enum
from typing import Generic, cast

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module import (
    AbstractDataModule,
)
from graph_mixup.augmentations.data.typing import (
    DataModuleConfigType,
)
from graph_mixup.augmentations.data.utils import initialize_transforms


class AbstractAugDataModule(AbstractDataModule, Generic[DataModuleConfigType]):
    def __init__(self, config: DataModuleConfigType) -> None:
        self.save_hyperparameters(dict(data_module_config=asdict(config)))
        self.config = config

        self.val_set: None | Dataset = None
        self.train_set: None | Dataset = None
        self.test_set: None | Dataset = None

        # Parse pre-computed dataset stats.
        with open(
            osp.join(
                osp.dirname(osp.realpath(__file__)), "dataset_statistics.json"
            ),
            "r",
        ) as f:
            all_dataset_stats: dict = json.load(f)
        dataset_stats: dict = all_dataset_stats[self.config.dataset_name]
        self.transforms, num_features, num_classes = initialize_transforms(
            dataset_stats
        )

        # Initialize parent class.
        super().__init__(
            config, num_features, num_classes, dataset_stats["num_graphs"]
        )

    @abstractmethod
    def transform_train_set(self, train_set: Dataset) -> Dataset:
        """Transform the vanilla train set into a mixup dataset."""

    @override
    def prepare_data(self) -> None:
        TUDataset(
            self.config.data_dir, cast(Enum, self.config.dataset_name).value
        )

    @override
    def setup(self, stage: str) -> None:
        dataset = TUDataset(
            self.config.data_dir,
            self.config.dataset_name,
            transform=self.transforms,
            use_node_attr=True,
        )

        dataset_idx = np.arange(len(dataset))
        dataset_labels = dataset.y.numpy()

        # Cross-validation
        skf = StratifiedKFold(
            self.config.num_outer_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        cv_splits = list(skf.split(dataset_idx, dataset_labels))
        cv_train_idx, cv_test_idx = cv_splits[self.config.fold]

        if stage == "fit":
            train_set_labels = dataset_labels[cv_train_idx]
            if not self.config.eval_mode:
                #
                # Holdout Validation in Model Selection:
                # Use the same training and validation sets throughout.
                #
                holdout_train_idx, holdout_val_idx = train_test_split(
                    cv_train_idx,
                    test_size=self.config.val_size,
                    # TODO: replace with inner holdout size
                    random_state=self.config.random_state,
                    stratify=train_set_labels,
                )
                self.log_cv_indices(holdout_train_idx, "holdout_train")
                self.log_cv_indices(holdout_val_idx, "holdout_val")

            else:
                #
                # Holdout Validation in Model Assessment:
                # Shuffle training and validation data (random_state + 1).
                #
                holdout_train_idx, holdout_val_idx = train_test_split(
                    cv_train_idx,
                    test_size=self.config.val_size,
                    # TODO: replace with inner holdout size
                    random_state=self.config.random_state + 1,
                    stratify=train_set_labels,
                )
                self.log_cv_indices(holdout_train_idx, "holdout_train_eval")
                self.log_cv_indices(holdout_val_idx, "holdout_val_eval")

            self.train_set = self.transform_train_set(
                dataset[holdout_train_idx]
            )
            self.val_set = dataset[holdout_val_idx]

        elif stage == "test":
            self.log_cv_indices(cv_test_idx, "test")
            self.test_set = dataset[cv_test_idx]
