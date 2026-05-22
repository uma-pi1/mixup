from graph_exporter.typing import IfMixupConfig

from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.search_spaces import SearchSpace, CategoricalSearchSpace


class IfMixupMethodHPProvider(AbstractMixupMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        mixup_alpha=CategoricalSearchSpace(
            "mixup_alpha", [0.1, 0.3, 0.5, 1.0, 5.0]
        ),
    )

    def get_mixup_method_config(self) -> IfMixupConfig:
        return IfMixupConfig(
            seed=0, mixup_alpha=self._hyperparams["mixup_alpha"]
        )
