from graph_mixup.augmentations.hp_provider import AbstractHPProvider
from graph_mixup.mixup_generation.emb_mixup.typing import (
    EmbMixupModelConfig,
)
from graph_mixup.search_spaces import (
    SearchSpace,
    FloatSearchSpace,
    CategoricalSearchSpace,
)


class EmbMixupModelHPProvider(AbstractHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        mixup_alpha=FloatSearchSpace("mixup_alpha", 0.05, 10.0),
        use_vanilla=CategoricalSearchSpace("use_vanilla", [True, False]),
        augmented_ratio=FloatSearchSpace("augmented_ratio", 0.2, 3.0),
    )

    def get_model_mixup_config(self) -> EmbMixupModelConfig:
        return EmbMixupModelConfig(
            mixup_alpha=self._hyperparams["mixup_alpha"],
            use_vanilla=self._hyperparams["use_vanilla"],
            augmented_ratio=self._hyperparams["augmented_ratio"],
        )
