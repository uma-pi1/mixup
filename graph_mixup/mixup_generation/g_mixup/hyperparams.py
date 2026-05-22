from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.mixup_generation.g_mixup.typing import GMixupConfig


class GMixupMethodHPProvider(AbstractMixupMethodHPProvider):
    def get_mixup_method_config(self) -> GMixupConfig:
        return GMixupConfig(
            seed=0, mixup_alpha=self._hyperparams["mixup_alpha"]
        )
