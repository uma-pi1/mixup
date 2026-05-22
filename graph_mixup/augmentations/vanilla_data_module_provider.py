from typing import final

from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module_provider import (
    AbstractDataModuleProvider,
)
from graph_mixup.augmentations.data.rdb_data_module import RDBDataModule
from graph_mixup.augmentations.data.typing import (
    NopDatasetMethodConfig,
    DatasetConfig,
    DataModuleConfig,
)


@final
class VanillaDataModuleProvider(AbstractDataModuleProvider):
    """Produces a DataModule of a dataset that does not apply any mixup/augmentation."""

    @override
    def _get_method_config(self) -> NopDatasetMethodConfig:
        return NopDatasetMethodConfig()

    @override
    def _get_dataset_config(
        self,
    ) -> DatasetConfig[NopDatasetMethodConfig]:
        return DatasetConfig(self._get_method_config())

    @override
    def _get_data_module_config(
        self,
    ) -> DataModuleConfig[DatasetConfig[NopDatasetMethodConfig]]:
        return DataModuleConfig(
            dataset_config=self._get_dataset_config(),
            **self._get_data_module_config_base_params(),
        )

    @override
    def get_data_module(
        self,
        inner_fold_idx: int | None,
        keep_sampled_mixup_graphs_ids: bool,
    ) -> RDBDataModule[DataModuleConfig[DatasetConfig[NopDatasetMethodConfig]]]:
        assert (
            keep_sampled_mixup_graphs_ids is False
        ), "VanillaDataModuleProvider does not use mixup graphs."

        return RDBDataModule(
            config=self._get_data_module_config(),
            method_name=None,
            inner_fold=inner_fold_idx,
            keep_sampled_mixup_graph_ids=keep_sampled_mixup_graphs_ids,
        )
