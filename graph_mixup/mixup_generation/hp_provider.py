from abc import ABC, abstractmethod
from typing import Any


from graph_mixup.augmentations.data.typing import (
    AbstractDatasetMethodConfig,
)
from graph_mixup.augmentations.hp_provider import AbstractHPProvider
from graph_mixup.search_spaces import (
    SearchSpace,
    FloatSearchSpace,
)


class AbstractMixupMethodHPProvider(AbstractHPProvider, ABC):
    search_spaces: dict[str, SearchSpace] = dict(
        mixup_alpha=FloatSearchSpace("mixup_alpha", 0.05, 10.0),
    )

    def _get_base_config(self) -> dict[str, Any]:
        return dict(
            mixup_alpha=self._hyperparams["mixup_alpha"],
        )

    @abstractmethod
    def get_mixup_method_config(self) -> AbstractDatasetMethodConfig: ...
