from typing_extensions import override

from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.ged_mixup.typing import GEDMixupMethodConfig
from graph_mixup.search_spaces import SearchSpace, CategoricalSearchSpace


class GEDMixupMethodHPProvider(AbstractMixupMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        mixup_alpha=CategoricalSearchSpace(
            "mixup_alpha", [0.1, 0.3, 0.5, 1.0, 5.0]
        ),
    )

    @override
    def get_mixup_method_config(self) -> GEDMixupMethodConfig:
        return GEDMixupMethodConfig(
            mixup_alpha=self._hyperparams["mixup_alpha"],
            max_items_per_pair=1,
        )
