from abc import ABC, abstractmethod
from typing import Any

import yaml
from optuna import Trial
from typing_extensions import override, final

from graph_mixup.augmentations.hp_provider import AbstractHPProvider
from graph_mixup.config.typing import CLConfig
from graph_mixup.models.typing import (
    GNNParams,
    AbstractModelMethodConfig,
    OptimizerName,
    AggrName,
    NormName,
)
from graph_mixup.resource_locators import BaselineResourceLocator
from graph_mixup.search_spaces import (
    CategoricalSearchSpace,
    FloatSearchSpace,
    IntSearchSpace,
    SearchSpace,
)


class GNNHPProvider(AbstractHPProvider, ABC):
    def __init__(self, trial: Trial, config: CLConfig) -> None:
        super().__init__(trial)
        self.config = config

    @abstractmethod
    def get_params(
        self, method_config: AbstractModelMethodConfig
    ) -> GNNParams: ...


@final
class SuggestGNNHPProvider(GNNHPProvider):
    search_spaces: dict[str, SearchSpace] = dict(
        optimizer=CategoricalSearchSpace(
            "optimizer", OptimizerName.get_member_names()
        ),
        lr=FloatSearchSpace("lr", 1e-5, 1e-1, log=True),
        num_conv_layers=IntSearchSpace("num_conv_layers", 2, 8, step=2),
        hidden_channels=IntSearchSpace("hidden_channels", 32, 256, log=True),
        aggr=CategoricalSearchSpace("aggr", AggrName.get_member_names()),
        dropout=FloatSearchSpace("dropout", 0.0, 0.5),
        norm=CategoricalSearchSpace(
            "norm", [None] + NormName.get_member_names()
        ),
        num_pre_processing_layers=IntSearchSpace(
            "num_pre_processing_layers", 1, 3
        ),
        num_post_processing_layers=IntSearchSpace(
            "num_post_processing_layers", 1, 3
        ),
    )

    def _get_base_params(self) -> dict[str, Any]:
        return self._hyperparams | dict(
            optimizer=OptimizerName[self._hyperparams["optimizer"]],
            aggr=AggrName[self._hyperparams["aggr"]],
            norm=(
                NormName[self._hyperparams["norm"]]
                if self._hyperparams["norm"] is not None
                else None
            ),
            seed=self.config.seed,
        )

    @override
    def get_params(self, method_config: AbstractModelMethodConfig) -> GNNParams:
        return GNNParams(
            method_config=method_config,
            **self._get_base_params(),
        )


@final
class LoadGNNHPProvider(GNNHPProvider):
    search_spaces: dict[str, SearchSpace] = dict()

    # noinspection PyTypeChecker
    @override
    def get_params(self, method_config: AbstractModelMethodConfig) -> GNNParams:
        locator = BaselineResourceLocator(self.config)
        best_params_path = locator.get_best_trial_file_path()

        with open(best_params_path, "r") as f:
            best_params_file_dict = yaml.safe_load(f)

        params_dict = best_params_file_dict["model_selection"]["params"]

        return GNNParams(
            optimizer=OptimizerName[params_dict["optimizer"]],
            lr=params_dict["lr"],
            num_conv_layers=params_dict["num_conv_layers"],
            hidden_channels=params_dict["hidden_channels"],
            aggr=AggrName[params_dict["aggr"]],
            dropout=params_dict["dropout"],
            norm=(
                NormName[params_dict["norm"]]
                if params_dict["norm"] is not None
                else None
            ),
            num_pre_processing_layers=params_dict["num_pre_processing_layers"],
            num_post_processing_layers=params_dict[
                "num_post_processing_layers"
            ],
            method_config=method_config,
            seed=self.config.seed,
        )


@final
class GNNHPProviderFactory:
    @staticmethod
    def get_provider(config: CLConfig, trial: Trial) -> GNNHPProvider:
        """
        Returns an HP-Provider that either loads HPs from the baseline or asks
        for new suggestions from optuna.

        This depends on the config: If the config specifies to use baseline
        params, then baseline params will be loaded. This only works if a method
        is specified (if there is no method, then there cannot be a baseline
        since "no method" is equal to "baseline").

        Otherwise, receives parameter suggestions from optuna.
        """
        if not config.use_baseline or config.method_name is None:
            return SuggestGNNHPProvider(trial, config)
        else:
            return LoadGNNHPProvider(trial, config)
