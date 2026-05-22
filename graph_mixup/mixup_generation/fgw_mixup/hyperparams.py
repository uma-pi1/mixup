from graph_exporter.typing import FGWMixupConfig

from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.search_spaces import (
    SearchSpace,
    CategoricalSearchSpace,
)


class FGWMixupMethodHPProvider(AbstractMixupMethodHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        fgw_alpha=CategoricalSearchSpace("fgw_alpha", [0.05, 0.5, 0.95]),
        rho=CategoricalSearchSpace("rho", [0.1, 1, 10]),
    )

    def get_mixup_method_config(self) -> FGWMixupConfig:
        return FGWMixupConfig(
            seed=0,
            fgw_alpha=self._hyperparams["fgw_alpha"],
            rho=self._hyperparams["rho"],
            measure="uniform",
            metric="adj",
            mixup_alpha=0.2,
            loss_fun="square_loss",
        )
