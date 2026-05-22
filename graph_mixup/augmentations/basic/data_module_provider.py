from typing import assert_never

from graph_mixup.augmentations.basic.data_module import BasicAugDataModule
from graph_mixup.augmentations.basic.hyperparams import (
    DropEdgeHPProvider,
    DropNodeHPProvider,
    DropPathHPProvider,
    PerturbNodeAttrHPProvider,
)
from graph_mixup.augmentations.basic.typing import BasicAugDatasetMethodConfig
from graph_mixup.augmentations.data.abstract_aug_data_module import (
    AbstractAugDataModule,
)
from graph_mixup.augmentations.data.abstract_data_module_provider import (
    AbstractDataModuleProvider,
)
from graph_mixup.augmentations.data.typing import (
    DataModuleConfig,
)
from graph_mixup.augmentations.hp_provider import AugDatasetHPProvider
from graph_mixup.augmentations.typing import AugDatasetConfig
from graph_mixup.config.typing import AugmentationName


class BasicAugDataModuleProvider(AbstractDataModuleProvider):
    def _get_method_config(self) -> BasicAugDatasetMethodConfig:
        if not isinstance(self.config.method_name, AugmentationName):
            raise TypeError(
                self.__class__.__name__
                + " can only be used for basic augmentations"
            )

        if self.config.method_name is AugmentationName.DROP_EDGE:
            return DropEdgeHPProvider(self.trial).get_method_config()

        if self.config.method_name is AugmentationName.DROP_NODE:
            return DropNodeHPProvider(self.trial).get_method_config()

        if self.config.method_name is AugmentationName.DROP_PATH:
            return DropPathHPProvider(self.trial).get_method_config()

        if self.config.method_name is AugmentationName.PERTURB_NODE_ATTR:
            return PerturbNodeAttrHPProvider(self.trial).get_method_config()

        assert_never(self.config.method_name)

    def _get_dataset_config(
        self,
    ) -> AugDatasetConfig[BasicAugDatasetMethodConfig]:
        return AugDatasetHPProvider(self.trial).get_dataset_config(
            self._get_method_config()
        )

    def _get_data_module_config(
        self,
    ) -> DataModuleConfig[AugDatasetConfig[BasicAugDatasetMethodConfig]]:
        return DataModuleConfig(
            dataset_config=self._get_dataset_config(),
            **self._get_data_module_config_base_params(),
        )

    def get_data_module(self) -> AbstractAugDataModule:
        return BasicAugDataModule(self._get_data_module_config())
