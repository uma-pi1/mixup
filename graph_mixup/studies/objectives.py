import logging
import math
from abc import ABC, abstractmethod
from datetime import timedelta
from statistics import mean
from typing import Iterable, Union

import numpy as np
from graph_exporter.typing import MixupItem
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from optuna import Trial, TrialPruned
from optuna.trial import BaseTrial, FrozenTrial
from typing_extensions import assert_never, override

from graph_mixup.augmentations.data.abstract_data_module import (
    AbstractDataModule,
)
from graph_mixup.augmentations.data.rdb_data_module import RDBDataModule
from graph_mixup.augmentations.typing import AugDatasetConfig
from graph_mixup.augmentations.vanilla_data_module_provider import (
    VanillaDataModuleProvider,
)
from graph_mixup.config.typing import (
    AugmentationName,
    CLConfig,
    PreBatchMixupName,
    PreLossMixupName,
)
from graph_mixup.ged_database.handlers.graph_import_database_handler import (
    GraphImportDatabaseHandler,
)
from graph_mixup.ged_mixup.data_module_provider import (
    GEDMixupDataModuleProvider,
)
from graph_mixup.mixup_generation.fgw_mixup.data_module_provider import (
    FGWMixupDataModuleProvider,
)
from graph_mixup.mixup_generation.g_mixup.data_module_provider import (
    GMixupDataModuleProvider,
)
from graph_mixup.mixup_generation.geomix.data_module_provider import (
    GeoMixDataModuleProvider,
    GeoMixSEDataModuleProvider,
)
from graph_mixup.mixup_generation.if_mixup.data_module_provider import (
    IfMixupDataModuleProvider,
    IfMixupSEDataModuleProvider,
)
from graph_mixup.mixup_generation.s_mixup.data_module_provider import (
    SMixupDataModuleProvider,
)
from graph_mixup.mixup_generation.submix.data_module_provider import (
    SubMixDataModuleProvider,
)
from graph_mixup.models.lightning_gnn import LitGNN
from graph_mixup.models.model_provider import ModelProvider
from graph_mixup.models.typing import ModelMixupConfig
from graph_mixup.resource_locators import ResourceLocator
from graph_mixup.studies.loggers import InMemoryLogger
from graph_mixup.typing import TestResult

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _safe_mean(values: list[float | None]) -> float:
    # Handle possible None values (esp. when all values are None).
    filtered_values = [v for v in values if v is not None]
    return mean(filtered_values) if filtered_values else 0.0


def _create_jagged_nan_array(
    index_value_maps: list[dict[int, float]],
) -> np.ndarray:
    max_len = max(len(map) for map in index_value_maps)

    # Check whether indices match.
    indices = [list(map.keys()) for map in index_value_maps]
    for i in range(max_len):
        i_indices: list[int] = []
        for arr in indices:
            if i < len(arr):
                i_indices.append(arr[i])
        assert len(set(i_indices)) == 1, "Indices do not match across maps."

    # Create combined array (filled with NaNs when lengths differ).
    combined = np.full((len(index_value_maps), max_len), np.nan)
    for i, map in enumerate(index_value_maps):
        combined[i, : len(map)] = list(map.values())
    return combined


def _get_best_average_value_and_index(
    index_value_maps: list[dict[int, float]], *, max: bool
) -> tuple[float, int]:
    combined = _create_jagged_nan_array(index_value_maps)

    # Compute row-wise averages.
    averages = np.nanmean(combined, axis=0)

    # Compute best average value and its associated relative index.
    if max:
        best_avg_value = np.max(averages)
        best_avg_value_rel_idx = np.argmax(averages)
    else:
        best_avg_value = np.min(averages)
        best_avg_value_rel_idx = np.argmin(averages)

    # Convert relative index to index of original maps.
    max_len_idx = np.argmax([len(map) for map in index_value_maps])
    best_avg_value_index = list(index_value_maps[max_len_idx].keys())[
        best_avg_value_rel_idx
    ]

    return best_avg_value, best_avg_value_index


class LogEveryNthEpoch(Callback):
    def __init__(self, n: int = 100) -> None:
        super().__init__()
        self.n = n

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.n == 0:
            logger.info(f"Epoch {trainer.current_epoch + 1} complete.")


class BaseObjective(ABC):
    def __init__(self, config: CLConfig):
        self.config = config
        self.locator = ResourceLocator(config)

    @abstractmethod
    def __call__(
        self, trial: Trial | FrozenTrial
    ) -> Union[float, TestResult]: ...

    def _create_trainer(
        self, trial: BaseTrial, version_identifier: str, *, eval_mode: bool
    ) -> Trainer:
        callbacks = [LogEveryNthEpoch()]

        if eval_mode:
            # ===
            # n_epochs has been determined during model selection. No early
            # stopping required.
            # ===
            assert "best_avg_val_acc_epoch" in trial.user_attrs, (
                "n_epochs must be determined during model selection (expecting "
                "trial attr 'best_avg_val_acc_epoch'."
            )
            max_epochs = trial.user_attrs["best_avg_val_acc_epoch"]
        else:
            # ===
            # n_epochs unknown (i.e., currently in model selection). Early
            # Stopping beneficial.
            # ===
            max_epochs = self.config.max_epochs
            callbacks.append(
                EarlyStopping(
                    monitor="val_acc",
                    mode="max",
                    patience=self.config.patience,
                )
            )

        return Trainer(
            logger=self._create_loggers(version_identifier),
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_checkpointing=False,
            reload_dataloaders_every_n_epochs=self.config.reload_dataloaders_every_n_epochs,
            max_time=(
                timedelta(seconds=self.config.train_timeout)
                if self.config.train_timeout is not None
                else None
            ),
            devices=[self.config.device],
            check_val_every_n_epoch=20,
            callbacks=callbacks,
            deterministic=False,
            enable_model_summary=False,
            log_every_n_steps=100,
        )

    def _create_loggers(
        self, version_identifier: str
    ) -> Logger | Iterable[Logger]:
        return InMemoryLogger(), TensorBoardLogger(
            save_dir=self.locator.get_experiment_log_dir_path(),
            name=self.locator.get_fold_name(),
            version=version_identifier,
        )

    def _create_data_module(
        self,
        trial: BaseTrial,
        eval_mode: bool,
        inner_fold: int | None = None,
        *,
        keep_sampled_mixup_graph_ids: bool = False,
    ) -> RDBDataModule:
        assert eval_mode or inner_fold is not None

        match self.config.method_name:
            # ===
            # RDB Data Modules:
            # ===

            case None:
                return VanillaDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.FGW_MIXUP:
                return FGWMixupDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.GEOMIX:
                return GeoMixDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.GEOMIX_SE:
                return GeoMixSEDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.IF_MIXUP:
                return IfMixupDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.IF_MIXUP_SE:
                return IfMixupSEDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.GED_MIXUP:
                return GEDMixupDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.SUBMIX:
                return SubMixDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreLossMixupName.EMB_MIXUP:
                return VanillaDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.G_MIXUP:
                return GMixupDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case PreBatchMixupName.S_MIXUP:
                return SMixupDataModuleProvider(
                    self.config, eval_mode, trial
                ).get_data_module(inner_fold, keep_sampled_mixup_graph_ids)

            case _:
                assert_never(self.config.method_name)

    @abstractmethod
    def _create_model(
        self, trial: Trial, num_features: int, num_classes: int
    ) -> LitGNN: ...


class TrainObjective(BaseObjective):
    @override
    def __call__(self, trial: Trial) -> float:
        epochwise_train_losses: list[dict[int, float]] = []
        epochwise_val_accs: list[dict[int, float]] = []
        epochwise_val_losses: list[dict[int, float]] = []
        for inner_fold in range(self.config.num_inner_folds):
            logger.info(f"Starting inner fold {inner_fold}.")

            # Get data module.
            data_module = self._create_data_module(trial, False, inner_fold)

            # Get model.
            model = self._create_model(
                trial, data_module.num_features, data_module.num_classes
            )

            # Conduct sanity check.
            if not self._sanity_check_params(
                dataset_len=data_module.num_graphs,
                batch_size=data_module.config.batch_size,
                use_vanilla=self._resolve_use_vanilla(data_module, model),
                augmented_ratio=self._resolve_augmented_ratio(
                    data_module, model
                ),
                uses_batch_norm=model.uses_batch_norm,
            ):
                exception = TrialPruned()
                note = "PRUNED TRIAL: setup did not pass sanity check."
                print(note)
                exception.add_note(note)
                raise exception

            # Train.
            trainer = self._create_trainer(
                trial,
                f"trial_{trial.number}-inner_fold_{inner_fold}",
                eval_mode=False,
            )
            trainer.fit(model, datamodule=data_module)

            # Extract metrics.
            memory_logger = trainer.logger
            assert isinstance(memory_logger, InMemoryLogger)
            epochwise_val_accs.append(
                memory_logger.get_epochwise_metric("val_acc")
            )
            epochwise_val_losses.append(
                memory_logger.get_epochwise_metric("val_loss")
            )
            epochwise_train_losses.append(
                memory_logger.get_epochwise_metric("train_loss_epoch")
            )

            if self.config.use_inner_holdout:
                # ===
                # Inner Holdout Validation: Only use a single train+val split
                # during model selection.
                # ===
                break

        # ===
        # Log and report: Epoch-wise metrics.
        # ===

        trial.set_user_attr("val_accs", epochwise_val_accs)
        trial.set_user_attr("val_losses", epochwise_val_losses)
        trial.set_user_attr("train_losses_epoch", epochwise_train_losses)

        # ===
        # Log and report: Summarized metrics.
        # ===

        best_avg_val_acc, best_avg_val_acc_epoch = (
            _get_best_average_value_and_index(epochwise_val_accs, max=True)
        )
        best_avg_val_loss, best_avg_val_loss_epoch = (
            _get_best_average_value_and_index(epochwise_val_losses, max=False)
        )
        best_avg_train_loss, best_avg_train_loss_epoch = (
            _get_best_average_value_and_index(epochwise_train_losses, max=False)
        )
        trial.set_user_attr("best_avg_val_acc", best_avg_val_acc)
        trial.set_user_attr(
            "best_avg_val_acc_epoch", best_avg_val_acc_epoch + 1
        )
        trial.set_user_attr("best_avg_val_loss", best_avg_val_loss)
        trial.set_user_attr(
            "best_avg_val_loss_epoch", best_avg_val_loss_epoch + 1
        )
        trial.set_user_attr("best_avg_train_loss", best_avg_train_loss)
        trial.set_user_attr(
            "best_avg_train_loss_epoch", best_avg_train_loss_epoch + 1
        )

        # ===
        # For logging purposes: Compute test metrics.
        # ===

        if self.config.test_every_trial:
            logger.info("Compute test metrics (for logging).")

            # Hack: Convert current trial into a frozen one.
            frozen_trial = trial.study.trials[trial.number]

            # Use all available training data of current fold (eval_mode).
            data_module = self._create_data_module(frozen_trial, True)

            model = self._create_model(
                frozen_trial, data_module.num_features, data_module.num_classes
            )

            # Use the best epoch for evaluation (eval_mode).
            trainer = self._create_trainer(
                frozen_trial,
                f"trial_{frozen_trial.number}-test",
                eval_mode=True,
            )

            # ===
            # Train and extract metrics.
            # ===

            trainer.fit(model, datamodule=data_module)

            memory_logger = trainer.logger
            assert isinstance(memory_logger, InMemoryLogger)

            test_final_train_loss = memory_logger.get_metric_values(
                "train_loss_epoch"
            )[-1]

            # ===
            # Test and extract metrics.
            # ===

            trainer.test(model, datamodule=data_module, verbose=False)

            test_acc = memory_logger.get_metric_value("test_acc")
            test_loss = memory_logger.get_metric_value("test_loss")

            # ===
            # Store test metrics in current (non-frozen) trial for logging purposes.
            # ===

            trial.set_user_attr(
                "test_result",
                dict(
                    test_final_train_loss=test_final_train_loss
                    if not math.isnan(test_final_train_loss)
                    else None,
                    test_acc=test_acc,
                    test_loss=test_loss if not math.isnan(test_loss) else None,
                    test_epochs=trainer.max_epochs,
                ),
            )

        # ===
        # For HPO: Use validation metrics (does not touch test scores!).
        # ===

        return best_avg_val_acc

    def _sanity_check_params(
        self,
        *,
        dataset_len: int,
        batch_size: int,
        use_vanilla: bool,
        augmented_ratio: float,
        uses_batch_norm: bool,
    ) -> bool:
        if self.config.method_name is PreLossMixupName.EMB_MIXUP:
            # Check whether there are sufficient items after mixup (esp. when
            # batch norm is used).
            batch_size_after_mixup = int(
                (1 + augmented_ratio) * batch_size
                if use_vanilla
                else augmented_ratio * batch_size
            )

            required_batch_size = 2 if uses_batch_norm else 1

            return batch_size_after_mixup >= required_batch_size
        else:
            # Check whether there is at least one batch in the train loader.
            train_size = int(
                dataset_len
                * (self.config.num_outer_folds - 1)
                / self.config.num_outer_folds  # gives train+val set size
                * (self.config.num_inner_folds - 1)
                / self.config.num_inner_folds  # gives train set size
            )
            augmented_size = int(
                (1 + augmented_ratio) * train_size
                if use_vanilla
                else augmented_ratio * train_size
            )
            num_batches = augmented_size // batch_size
            return num_batches > 0

    def _resolve_use_vanilla(
        self, data_module: AbstractDataModule, model: LitGNN
    ) -> bool:
        if self.config.method_name is None:
            return True

        if isinstance(self.config.method_name, PreLossMixupName):
            if isinstance(model.method_config, ModelMixupConfig):
                return model.method_config.use_vanilla

            raise TypeError(
                "pre-loss mixup methods require 'use_vanilla' to be set inside the model method's config"
            )

        if isinstance(
            self.config.method_name, (PreBatchMixupName, AugmentationName)
        ):
            if isinstance(data_module.config.dataset_config, AugDatasetConfig):
                return data_module.config.dataset_config.use_vanilla

            raise TypeError(
                "pre-batch mixup methods or augmentation methods require 'use_vanilla' to be set inside the dataset's config"
            )

        assert_never(self.config.method_name)

    def _resolve_augmented_ratio(
        self, data_module: AbstractDataModule, model: LitGNN
    ) -> float:
        if self.config.method_name is None:
            return 0.0

        if isinstance(self.config.method_name, PreLossMixupName):
            if isinstance(model.method_config, ModelMixupConfig):
                return model.method_config.augmented_ratio

            raise TypeError(
                "pre-loss mixup methods require 'augmented_ratio' to be set inside the model method's config"
            )

        if isinstance(
            self.config.method_name, (PreBatchMixupName, AugmentationName)
        ):
            if isinstance(data_module.config.dataset_config, AugDatasetConfig):
                return data_module.config.dataset_config.augmented_ratio

            raise TypeError(
                "pre-batch mixup methods or augmentation methods require 'augmented_ratio' to be set inside the dataset's config"
            )

        assert_never(self.config.method_name)

    def _create_model(
        self, trial: BaseTrial, num_features: int, num_classes: int
    ) -> LitGNN:
        provider = ModelProvider(self.config, trial)
        return provider.get_model(num_features, num_classes, None)


class TestObjective(BaseObjective):
    def __init__(self, config: CLConfig, test_round: int) -> None:
        super().__init__(config)
        self.test_round = test_round

    def __call__(self, trial: Trial) -> TestResult:
        logger.info(f"Starting model assessment round {self.test_round}.")

        # Storing mixup graphs is supported for these methods.
        keep_sampled_mixup_graph_ids = self.config.method_name in [
            PreBatchMixupName.IF_MIXUP,
            PreBatchMixupName.IF_MIXUP_SE,
            PreBatchMixupName.GEOMIX,
            PreBatchMixupName.GEOMIX_SE,
            PreBatchMixupName.GED_MIXUP,
            PreBatchMixupName.S_MIXUP,
            PreBatchMixupName.FGW_MIXUP,
            PreBatchMixupName.SUBMIX,
        ]

        # Get data module.
        data_module = self._create_data_module(
            trial,
            True,
            keep_sampled_mixup_graph_ids=keep_sampled_mixup_graph_ids,
        )

        # Get model.
        model = self._create_model(
            trial, data_module.num_features, data_module.num_classes
        )

        # Train.
        trainer = self._create_trainer(
            trial, f"test_round_{self.test_round}", eval_mode=True
        )
        trainer.fit(model, datamodule=data_module)

        # Extract fit metrics.
        memory_logger = trainer.logger
        assert isinstance(memory_logger, InMemoryLogger)
        final_train_loss = memory_logger.get_metric_values("train_loss_epoch")[
            -1
        ]

        # Evaluate.
        trainer.test(model, datamodule=data_module)

        # Extract test metrics.
        memory_logger = trainer.logger
        assert isinstance(memory_logger, InMemoryLogger)
        test_acc = memory_logger.get_metric_value("test_acc")
        test_loss = memory_logger.get_metric_value("test_loss")

        # Extract mixup graphs.
        if keep_sampled_mixup_graph_ids:
            if self.config.method_name is PreBatchMixupName.S_MIXUP:
                # ===
                # Graphs need to be stored in the DB first.
                # ===
                logger.info("Store S-Mixup graphs in the DB.")
                db_handler = GraphImportDatabaseHandler()
                db_graph_ids = db_handler.create_mixup_graphs(
                    [
                        MixupItem(
                            graph_dict=item.graph.to_dict(),
                            lam=item.lam,
                            source_indices=item.source_indices,
                            creation_time_us=item.creation_time_us,
                        )
                        for item in data_module.sampled_smixup_items
                    ],
                    config=data_module.config.dataset_config,  # TODO: This may be incorrect.
                    dataset_name=self.config.dataset_name,
                    method_name=self.config.method_name,
                    sample_edges=True,
                )
                logger.info("S-Mixup graphs stored in the DB.")

                mixup_graph_ids = db_graph_ids

            else:
                # ===
                # In this case, the graphs are already inside the DB.
                # ===
                mixup_graph_ids = data_module.sampled_mixup_graph_ids

        else:
            mixup_graph_ids = None

        return TestResult(
            _final_train_loss=final_train_loss,
            test_acc=test_acc,
            _test_loss=test_loss,
            mixup_graph_ids=mixup_graph_ids,
        )

    def _create_model(
        self,
        trial: Trial,
        num_features: int,
        num_classes: int,
    ) -> LitGNN:
        provider = ModelProvider(self.config, trial)
        return provider.get_model(num_features, num_classes, self.test_round)
