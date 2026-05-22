from graph_exporter.typing import SubMixConfig

from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.search_spaces import SearchSpace, CategoricalSearchSpace


class SubMixMethodHPProvider(AbstractMixupMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        aug_size=CategoricalSearchSpace("aug_size", [0.2, 0.4, 0.6]),
    )

    def get_mixup_method_config(self) -> SubMixConfig:
        return SubMixConfig(seed=0, aug_size=self._hyperparams["aug_size"])
