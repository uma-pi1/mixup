from abc import ABC, abstractmethod
from typing import Any, Generic, final

from graph_exporter.typing import BaseConfig
from optuna import Trial
from typing_extensions import override

from graph_mixup.augmentations.data.typing import (
    AbstractDatasetMethodConfig,
    MethodConfigType,
)
from graph_mixup.augmentations.typing import (
    AugDatasetConfig,
    AugDatasetConfigType,
    GEDFilterFlags,
)
from graph_mixup.search_spaces import (
    SearchSpace,
    CategoricalSearchSpace,
    FloatSearchSpace,
)


class AbstractHPProvider(ABC):
    search_spaces: dict[str, SearchSpace] = dict()

    def __init__(self, trial: Trial) -> None:
        self._hyperparams: dict[str, Any] = {}
        for name, search_space in self.search_spaces.items():
            self._hyperparams[name] = search_space.suggest(trial)


class AbstractAugDatasetHPProvider(
    AbstractHPProvider, ABC, Generic[MethodConfigType, AugDatasetConfigType]
):
    search_spaces: dict[str, SearchSpace] = dict(
        use_vanilla=CategoricalSearchSpace("use_vanilla", [True, False]),
        augmented_ratio=FloatSearchSpace("augmented_ratio", 0.2, 2.0),
    )

    def _get_base_config(self) -> dict[str, Any]:
        return dict(
            use_vanilla=self._hyperparams["use_vanilla"],
            augmented_ratio=self._hyperparams["augmented_ratio"],
            ged_filter_flags=GEDFilterFlags(
                max_ged_value=None,
                only_same_class=False,
                only_different_class=False,
                only_first_absolute_quintile=False,
                only_last_absolute_quintile=False,
                only_first_relative_quintile=False,
                only_last_relative_quintile=False,
            ),
        )

    @abstractmethod
    def get_dataset_config(
        self, method_config: MethodConfigType
    ) -> AugDatasetConfigType: ...


@final
class AugDatasetHPProvider(AbstractAugDatasetHPProvider):
    """
    DatasetHPProvider for a mixup / augmentation method dataset without any
    custom parameters. This means that the following dataset parameters are
    present:
    - use_vanilla
    - augmented_ratio
    """

    @override
    def get_dataset_config(
        self, method_config: AbstractDatasetMethodConfig | BaseConfig
    ) -> AugDatasetConfig:
        return AugDatasetConfig(
            method_config=method_config, **self._get_base_config()
        )
