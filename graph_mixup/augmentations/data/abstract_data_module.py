import os
import os.path as osp
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, final
import warnings
import logging

import numpy as np
import yaml
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing_extensions import override

from graph_mixup.augmentations.data.typing import (
    DataModuleConfigType,
)

warnings.filterwarnings(
    "ignore", message="The 'train_dataloader' does not have many workers.*"
)
warnings.filterwarnings(
    "ignore", message="The 'val_dataloader' does not have many workers.*"
)
warnings.filterwarnings(
    "ignore", message="The 'test_dataloader' does not have many workers.*"
)

logger = logging.getLogger(__name__)


class AbstractDataModule(
    ABC, LightningDataModule, Generic[DataModuleConfigType]
):
    def __init__(
        self,
        config: DataModuleConfigType,
        num_features: int,
        num_classes: int,
        num_graphs: int,
        eval_mode: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dict(data_module_config=asdict(config)))
        self.config = config

        self.num_features: int = num_features
        self.num_classes: int = num_classes
        self.num_graphs: int = num_graphs
        self.eval_mode: bool = eval_mode

        # Should be set in the setup method.
        self.val_set: None | Dataset | list[Data] = None
        self.train_set: None | Dataset | list[Data] = None
        self.test_set: None | Dataset | list[Data] = None

    @override
    @abstractmethod
    def setup(self, stage: str) -> None: ...

    @final
    def log_cv_indices(self, indices: np.ndarray, indices_type: str) -> None:
        path = osp.join(
            self.config.data_dir,
            self.config.dataset_name,
            "cv_splits",
            f"cv_seed_{self.config.random_state}-n_outer_folds_{self.config.num_outer_folds}-n_inner_folds_{self.config.num_inner_folds}",
            f"fold_{self.config.fold}",
        )
        os.makedirs(path, exist_ok=True)

        filename = f"{indices_type}.yaml"
        content = set(indices.tolist())

        if not osp.exists(osp.join(path, filename)):
            with open(osp.join(path, filename), "w") as f:
                yaml.dump(content, f, width=80, default_flow_style=True)
        else:
            with open(osp.join(path, filename), "r") as f:
                assert yaml.safe_load(f) == content, "CV split mismatch."

    @override
    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None and len(self.train_set) > 0
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )

    @final
    @override
    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None
        if not self.eval_mode:
            assert len(self.val_set) > 0

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    @final
    @override
    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None and len(self.test_set) > 0
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
