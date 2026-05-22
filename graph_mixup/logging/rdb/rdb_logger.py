import math
from dataclasses import asdict
from datetime import datetime

from optuna.trial import FrozenTrial, TrialState
from sqlalchemy import create_engine, insert, select, update
from typing_extensions import override

from graph_mixup.config.typing import CLConfig
from graph_mixup.logging.base_logger import BaseLogger
from graph_mixup.logging.rdb.database import (
    TrialMetric,
    best_study_params,
    experiments,
    metadata_obj,
    results,
    studies,
    test_time_graphs,
    trial_hyperparams,
    trial_metrics,
    trials,
)
from graph_mixup.logging.utils import get_experiment_db_url
from graph_mixup.typing import TestResult, TrialResult


def _none_if_inf_or_nan(value: float | None) -> float | None:
    if value is None:
        return None

    return None if math.isinf(value) or math.isnan(value) else value


class RDBLogger(BaseLogger):
    def __init__(
        self, config: CLConfig, commit_hash: str, optuna_study_id: int
    ) -> None:
        self.engine = create_engine(get_experiment_db_url(), pool_recycle=3600)
        metadata_obj.create_all(self.engine)
        super().__init__(config, commit_hash, optuna_study_id)

    @override
    def _create_or_fetch_experiment(
        self, config: CLConfig, commit_hash: str
    ) -> int:
        # Check if experiment already exists.
        with self.engine.begin() as connection:
            existing_result = connection.execute(
                select(experiments.c.id).where(
                    (experiments.c.model_name == config.model_name)
                    & (experiments.c.dataset_name == config.dataset_name)
                    & (experiments.c.method_name == config.method_name)
                    & (experiments.c.num_folds == config.num_outer_folds)
                    & (experiments.c.seed == config.seed)
                    & (experiments.c.cv_seed == config.cv_seed)
                    & (experiments.c.use_params_from == config.use_params_from),
                )
            ).fetchone()

        if existing_result:
            return existing_result[0]

        # Create new experiment otherwise.
        with self.engine.begin() as connection:
            result = connection.execute(
                insert(experiments).values(
                    start_date=datetime.now(),
                    model_name=config.model_name,
                    dataset_name=config.dataset_name,
                    method_name=config.method_name,
                    num_folds=config.num_outer_folds,
                    num_test_rounds=config.num_test_rounds,
                    seed=config.seed,
                    cv_seed=config.cv_seed,
                    config=asdict(config),
                    commit_hash=commit_hash,
                    use_params_from=config.use_params_from,
                )
            )

        return result.inserted_primary_key[0]  # type: ignore

    @override
    def _create_study(self, optuna_study_id: int, fold: int) -> int:
        with self.engine.begin() as connection:
            result = connection.execute(
                insert(studies).values(
                    optuna_study_id=optuna_study_id,
                    experiment_id=self.experiment_id,
                    start_date=datetime.now(),
                    fold=fold,
                )
            )
        return result.inserted_primary_key[0]

    @override
    def log_model_selection_result(
        self, best_trial_result: TrialResult, all_trials: list[FrozenTrial]
    ) -> None:
        with self.engine.begin() as connection:
            # Insert trial number of best trial into studies relation.
            connection.execute(
                update(studies)
                .where(studies.c.id == self.study_id)
                .values(
                    best_trial_number=best_trial_result.trial_number,
                    best_avg_train_loss=_none_if_inf_or_nan(
                        best_trial_result.best_avg_train_loss
                    ),
                    best_avg_train_loss_epoch=best_trial_result.best_avg_train_loss_epoch,
                    best_avg_val_acc=best_trial_result.best_avg_val_acc,
                    best_avg_val_acc_epoch=best_trial_result.best_avg_val_acc_epoch,
                    best_avg_val_loss=_none_if_inf_or_nan(
                        best_trial_result.best_avg_val_loss
                    ),
                    best_avg_val_loss_epoch=best_trial_result.best_avg_val_loss_epoch,
                )
            )

            # Insert hyperparameters of best trial.
            for key, value in best_trial_result.params.items():
                connection.execute(
                    insert(best_study_params).values(
                        study_id=self.study_id,
                        key=key,
                        value=value,
                    )
                )

        # ===
        # Log trials and associated results, metrics, and parameters.
        # ===

        for trial in all_trials:
            if trial.state is TrialState.COMPLETE:
                with self.engine.begin() as connection:
                    trial_id = connection.execute(
                        insert(trials).values(
                            study_id=self.study_id,
                            optuna_trial_id=trial._trial_id,
                            trial_number=trial.number,
                            duration=trial.duration.total_seconds(),
                            result=trial.value,
                        )
                    ).inserted_primary_key[0]

                    # ===
                    # Trial hyperparameters.
                    # ===

                    for key, value in trial.params.items():
                        connection.execute(
                            insert(trial_hyperparams).values(
                                trial_id=trial_id,
                                key=key,
                                value=value,
                            )
                        )

                    # ===
                    # Trial metrics: Training.
                    # ===

                    for inner_fold, val_accs in enumerate(
                        trial.user_attrs["val_accs"]
                    ):
                        for epoch, value in val_accs.items():
                            connection.execute(
                                insert(trial_metrics).values(
                                    trial_id=trial_id,
                                    inner_fold=inner_fold,
                                    epoch=int(epoch) + 1,
                                    key=TrialMetric.VAL_ACC,
                                    value=value,
                                )
                            )

                    for inner_fold, val_accs in enumerate(
                        trial.user_attrs["val_losses"]
                    ):
                        for epoch, value in val_accs.items():
                            if not math.isnan(value):
                                connection.execute(
                                    insert(trial_metrics).values(
                                        trial_id=trial_id,
                                        inner_fold=inner_fold,
                                        epoch=int(epoch) + 1,
                                        key=TrialMetric.VAL_LOSS,
                                        value=_none_if_inf_or_nan(value),
                                    )
                                )

                    for inner_fold, train_losses in enumerate(
                        trial.user_attrs["train_losses_epoch"]
                    ):
                        for epoch, value in train_losses.items():
                            connection.execute(
                                insert(trial_metrics).values(
                                    trial_id=trial_id,
                                    inner_fold=inner_fold,
                                    epoch=int(epoch) + 1,
                                    key=TrialMetric.TRAIN_LOSS_EPOCH,
                                    value=_none_if_inf_or_nan(value),
                                )
                            )

                    # ===
                    # Trial metrics: Testing.
                    # ===

                    if "test_result" in trial.user_attrs:
                        test_result = trial.user_attrs["test_result"]
                        connection.execute(
                            insert(trial_metrics).values(
                                [
                                    dict(
                                        trial_id=trial_id,
                                        inner_fold=None,
                                        epoch=test_result["test_epochs"],
                                        key=TrialMetric.TEST_TRAIN_LOSS_EPOCH,
                                        value=_none_if_inf_or_nan(
                                            test_result["test_final_train_loss"]
                                        ),
                                    ),
                                    dict(
                                        trial_id=trial_id,
                                        inner_fold=None,
                                        epoch=test_result["test_epochs"],
                                        key=TrialMetric.TEST_ACC,
                                        value=test_result["test_acc"],
                                    ),
                                    dict(
                                        trial_id=trial_id,
                                        inner_fold=None,
                                        epoch=test_result["test_epochs"],
                                        key=TrialMetric.TEST_LOSS,
                                        value=_none_if_inf_or_nan(
                                            test_result["test_loss"]
                                        ),
                                    ),
                                ]
                            )
                        )

    @override
    def log_model_assessment_result(self, test_result: TestResult) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(results).values(
                    study_id=self.study_id,
                    final_train_loss=_none_if_inf_or_nan(
                        test_result.final_train_loss
                    ),
                    test_acc=test_result.test_acc,
                    test_loss=_none_if_inf_or_nan(test_result.test_loss),
                    created_at=datetime.now(),
                )
            )

            if test_result.mixup_graph_ids is not None:
                connection.execute(
                    insert(test_time_graphs),
                    [
                        dict(study_id=self.study_id, graph_id=id)
                        for id in test_result.mixup_graph_ids
                    ],
                )

    @override
    def log_study_completed(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(studies)
                .where(studies.c.id == self.study_id)
                .values(
                    end_date=datetime.now(),
                    is_complete=True,
                )
            )

    @override
    def log_study_incomplete(self, e: Exception) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(studies)
                .where(studies.c.id == self.study_id)
                .values(
                    end_date=datetime.now(),
                    is_complete=False,
                    err_msg=str(e),
                )
            )

    @override
    def dispose(self) -> None:
        self.engine.dispose()
