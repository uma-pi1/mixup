from abc import ABC, abstractmethod

from typing_extensions import final, override

from graph_mixup.augmentations.basic.typing import (
    DropEdgeDatasetMethodConfig,
    BasicAugDatasetMethodConfig,
    DropNodeDatasetMethodConfig,
    DropPathDatasetMethodConfig,
    PerturbNodeAttrDatasetMethodConfig,
)
from graph_mixup.augmentations.hp_provider import AbstractHPProvider
from graph_mixup.search_spaces import (
    FloatSearchSpace,
    IntSearchSpace,
    SearchSpace,
)


class AbstractBasicAugDatasetMethodHPProvider(AbstractHPProvider, ABC):
    @abstractmethod
    def get_method_config(self) -> BasicAugDatasetMethodConfig: ...


@final
class DropEdgeHPProvider(AbstractBasicAugDatasetMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        p=FloatSearchSpace("p", 0.1, 0.5),
    )

    @override
    def get_method_config(self) -> DropEdgeDatasetMethodConfig:
        return DropEdgeDatasetMethodConfig(p=self._hyperparams["p"])


@final
class DropNodeHPProvider(AbstractBasicAugDatasetMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        p=FloatSearchSpace("p", 0.1, 0.5),
    )

    @override
    def get_method_config(self) -> DropNodeDatasetMethodConfig:
        return DropNodeDatasetMethodConfig(p=self._hyperparams["p"])


@final
class DropPathHPProvider(AbstractBasicAugDatasetMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        p=FloatSearchSpace("p", 0.1, 0.5),
        walks_per_node=IntSearchSpace("walks_per_node", 1, 5),
        walk_length=IntSearchSpace("walk_length", 1, 5),
    )

    @override
    def get_method_config(self) -> DropPathDatasetMethodConfig:
        return DropPathDatasetMethodConfig(
            p=self._hyperparams["p"],
            walks_per_node=self._hyperparams["walks_per_node"],
            walk_length=self._hyperparams["walk_length"],
        )


@final
class PerturbNodeAttrHPProvider(AbstractBasicAugDatasetMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        std=FloatSearchSpace("std", 0.1, 2.5),
    )

    @override
    def get_method_config(self) -> PerturbNodeAttrDatasetMethodConfig:
        return PerturbNodeAttrDatasetMethodConfig(std=self._hyperparams["std"])
