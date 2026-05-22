from abc import ABC, abstractmethod
from typing import Any

import yaml
from optuna import Trial

from graph_mixup.augmentations.data.abstract_aug_data_module import (
    AbstractAugDataModule,
)
from graph_mixup.augmentations.data.typing import (
    AbstractDatasetMethodConfig,
    DatasetConfig,
    DataModuleConfig,
)
from graph_mixup.config.typing import CLConfig
from graph_mixup.models.typing import BaselineParams, NopModelMethodConfig
from graph_mixup.resource_locators import BaselineResourceLocator


class AbstractDataModuleProvider(ABC):
    def __init__(self, config: CLConfig, eval_mode: bool, trial: Trial):
        self.config = config
        self.trial = trial
        self.eval_mode = eval_mode

    @abstractmethod
    def _get_method_config(self) -> AbstractDatasetMethodConfig: ...

    @abstractmethod
    def _get_dataset_config(self) -> DatasetConfig: ...

    @abstractmethod
    def _get_data_module_config(self) -> DataModuleConfig: ...

    @abstractmethod
    def get_data_module(
        self, inner_fold_idx: int | None, keep_sampled_mixup_graphs_ids: bool
    ) -> AbstractAugDataModule: ...

    def _get_data_module_config_base_params(self) -> dict[str, Any]:
        return dict(
            data_dir=self.config.data_dir,
            dataset_name=self.config.dataset_name,
            num_outer_folds=self.config.num_outer_folds,
            num_inner_folds=self.config.num_inner_folds,
            fold=self.config.fold,
            random_state=self.config.cv_seed,
            num_workers=self.config.num_workers,
            batch_size=self._get_batch_size(),
            eval_mode=self.eval_mode,
            device=self.config.device,
            label_corruption_prob=self.config.label_corruption_prob,
        )

    def _get_batch_size(self) -> int:
        if self.config.use_baseline:
            baseline_params = self._load_baseline_hparams()
            batch_size = baseline_params.batch_size
        else:
            batch_size = self.trial.suggest_categorical(
                "batch_size", [8, 16, 32, 64, 128, 256]
            )
        return batch_size

    def _load_baseline_hparams(self) -> BaselineParams:
        locator = BaselineResourceLocator(self.config)
        best_params_path = locator.get_best_trial_file_path()

        with open(best_params_path, "r") as f:
            best_params_file_dict = yaml.safe_load(f)

        return BaselineParams(
            method_config=NopModelMethodConfig(),
            seed=self.config.seed,
            **best_params_file_dict["model_selection"]["params"],
        )
