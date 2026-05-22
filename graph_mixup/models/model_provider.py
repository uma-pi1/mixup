from optuna import Trial

from graph_mixup.mixup_generation.emb_mixup.hp_provider import (
    EmbMixupModelHPProvider,
)
from graph_mixup.config.typing import CLConfig
from graph_mixup.models.hyperparams import GNNHPProviderFactory
from graph_mixup.models.lightning_gnn import LitGNN
from graph_mixup.models.typing import (
    GNNParams,
    AbstractModelMethodConfig,
    NopModelMethodConfig,
)


class ModelProvider:
    def __init__(self, config: CLConfig, trial: Trial) -> None:
        self.config = config
        self.trial = trial

    def _get_method_params(self) -> AbstractModelMethodConfig:
        if self.config.method_name == "emb_mixup":
            provider = EmbMixupModelHPProvider(self.trial)
            return provider.get_model_mixup_config()

        return NopModelMethodConfig()

    def _get_lit_gnn_params(self) -> GNNParams:
        method_config = self._get_method_params()

        gnn_hp_provider = GNNHPProviderFactory.get_provider(
            self.config, self.trial
        )

        return gnn_hp_provider.get_params(method_config)

    def get_model(
        self, num_features: int, num_classes: int, test_round: int | None
    ) -> LitGNN:
        return LitGNN(
            model=self.config.model_name,
            in_channels=num_features,
            out_channels=num_classes,
            gnn_params=self._get_lit_gnn_params(),
            test_round=test_round,
        )

    def load_model_from_ckpt(
        self, num_features: int, num_classes: int, ckpt_path: str
    ) -> LitGNN:
        return LitGNN.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=self.config.model_name,
            in_channels=num_features,
            out_channels=num_classes,
            gnn_params=self._get_lit_gnn_params(),
        )
