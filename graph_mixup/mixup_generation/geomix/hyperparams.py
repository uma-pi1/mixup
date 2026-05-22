from typing import Any

from graph_exporter.typing import GeoMixConfig

from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.search_spaces import (
    SearchSpace,
    CategoricalSearchSpace,
)


class GeoMixMethodHPProvider(AbstractMixupMethodHPProvider):
    search_spaces: dict[Any, SearchSpace] = dict(
        mixup_alpha=CategoricalSearchSpace("mixup_alpha", [0.1, 0.3, 0.5]),
    )

    def get_mixup_method_config(self) -> GeoMixConfig:
        return GeoMixConfig(
            seed=0,
            mixup_alpha=self._hyperparams["mixup_alpha"],
            num_graphs=10,
            num_nodes=20,
            alpha_fgw=1.0,
            sample_dist="beta",
            uniform_min=0.0,
            uniform_max=0.05,
            clip_eps=0.001,
        )
