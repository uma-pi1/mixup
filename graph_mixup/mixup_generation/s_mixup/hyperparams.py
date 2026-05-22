from optuna import Trial

from graph_mixup.augmentations.hp_provider import (
    AbstractAugDatasetHPProvider,
    AbstractHPProvider,
)
from graph_mixup.mixup_generation.hp_provider import (
    AbstractMixupMethodHPProvider,
)
from graph_mixup.mixup_generation.s_mixup.typing import (
    SMixupMethodConfig,
    SMixupDatasetConfig,
    LitGMNetConfig,
    LitGMNetTrainingConfig,
)
from graph_mixup.search_spaces import (
    CategoricalSearchSpace,
    SearchSpace,
    IntSearchSpace,
)

# ===
# GMNet Hyperparameters.
# ===


class LitGMnetHPProvider(AbstractHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        num_layers=IntSearchSpace("gmnet_num_layers", 4, 6),
    )

    def get_config(self) -> LitGMNetConfig:
        return LitGMNetConfig(num_layers=self._hyperparams["num_layers"])


class LitGMnetTrainingHPProvider(AbstractHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        batch_size=CategoricalSearchSpace("gmnet_batch_size", [8, 64, 128]),
    )

    def get_config(self):
        return LitGMNetTrainingConfig(
            batch_size=self._hyperparams["batch_size"],
        )


# ===
# S-Mixup Hyperparameters.
# ===


class SMixupMethodHPProvider(AbstractMixupMethodHPProvider):
    def get_mixup_method_config(self) -> SMixupMethodConfig:
        return SMixupMethodConfig(**self._get_base_config())


class SMixupDatasetHPProvider(AbstractAugDatasetHPProvider):
    def __init__(self, trial: Trial):
        super().__init__(trial)
        self.lit_gmnet_hp_provider = LitGMnetHPProvider(trial)
        self.lit_gmnet_training_hp_provider = LitGMnetTrainingHPProvider(trial)

    def get_dataset_config(
        self, method_config: SMixupMethodConfig
    ) -> SMixupDatasetConfig:
        return SMixupDatasetConfig(
            method_config=method_config,
            lit_gmnet_config=self.lit_gmnet_hp_provider.get_config(),
            lit_gmnet_training_config=self.lit_gmnet_training_hp_provider.get_config(),
            **self._get_base_config(),
        )
