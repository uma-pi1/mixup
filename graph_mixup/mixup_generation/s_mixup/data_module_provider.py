from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module_provider import (
    AbstractDataModuleProvider,
)
from graph_mixup.augmentations.data.rdb_data_module import RDBDataModule
from graph_mixup.mixup_generation.s_mixup.hyperparams import (
    SMixupMethodHPProvider,
    SMixupDatasetHPProvider,
)
from graph_mixup.mixup_generation.s_mixup.typing import (
    SMixupMethodConfig,
    SMixupDatasetConfig,
    SMixupDataModuleConfig,
)
from graph_mixup.config.typing import PreBatchMixupName


class SMixupDataModuleProvider(AbstractDataModuleProvider):
    @override
    def _get_method_config(self) -> SMixupMethodConfig:
        provider = SMixupMethodHPProvider(self.trial)
        return provider.get_mixup_method_config()

    @override
    def _get_dataset_config(self) -> SMixupDatasetConfig:
        provider = SMixupDatasetHPProvider(self.trial)
        return provider.get_dataset_config(self._get_method_config())

    @override
    def _get_data_module_config(self) -> SMixupDataModuleConfig:
        base_config_params = self._get_data_module_config_base_params()
        dataset_config = self._get_dataset_config()
        return SMixupDataModuleConfig(
            dataset_config=dataset_config,
            **base_config_params,
        )

    @override
    def get_data_module(
        self, inner_fold_idx: int | None, keep_sampled_mixup_graph_ids: bool
    ) -> RDBDataModule:
        return RDBDataModule(
            config=self._get_data_module_config(),
            method_name=PreBatchMixupName.S_MIXUP,
            inner_fold=inner_fold_idx,
            keep_sampled_mixup_graph_ids=keep_sampled_mixup_graph_ids,
        )
