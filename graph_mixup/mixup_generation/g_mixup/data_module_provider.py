from graph_exporter.typing import BaseConfig
from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module_provider import (
    AbstractDataModuleProvider,
)
from graph_mixup.augmentations.data.rdb_data_module import RDBDataModule
from graph_mixup.augmentations.data.typing import DataModuleConfig
from graph_mixup.augmentations.hp_provider import AugDatasetHPProvider
from graph_mixup.mixup_generation.typing import (
    MixupDatasetMethodConfig,
)
from graph_mixup.augmentations.typing import AugDatasetConfig
from graph_mixup.config.typing import PreBatchMixupName
from graph_mixup.mixup_generation.g_mixup.hyperparams import (
    GMixupMethodHPProvider,
)


class GMixupDataModuleProvider(AbstractDataModuleProvider):
    @override
    def _get_method_config(self) -> BaseConfig:
        method_hp_provider = GMixupMethodHPProvider(self.trial)
        return method_hp_provider.get_mixup_method_config()

    @override
    def _get_dataset_config(self) -> AugDatasetConfig[BaseConfig]:
        method_config = self._get_method_config()
        hp_provider = AugDatasetHPProvider(self.trial)
        return hp_provider.get_dataset_config(method_config)

    @override
    def _get_data_module_config(self) -> DataModuleConfig:
        base_config = self._get_data_module_config_base_params()

        return DataModuleConfig(
            dataset_config=(self._get_dataset_config()), **base_config
        )

    @override
    def get_data_module(
        self,
        inner_fold: int | None,
        keep_sampled_mixup_graphs_ids: bool,
    ) -> RDBDataModule[
        DataModuleConfig[AugDatasetConfig[MixupDatasetMethodConfig]]
    ]:
        return RDBDataModule(
            config=self._get_data_module_config(),
            method_name=PreBatchMixupName.G_MIXUP,
            inner_fold=inner_fold,
            keep_sampled_mixup_graph_ids=keep_sampled_mixup_graphs_ids,
        )
