from optuna import Study
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial

from graph_mixup.config.typing import CLConfig
from graph_mixup.studies.objectives import TrainObjective, TestObjective
from graph_mixup.studies.utils import create_optuna_study, load_optuna_study
from graph_mixup.typing import TrialResult, TestResult


class OneShotCallback:
    def __call__(self, study: Study, _: FrozenTrial) -> None:
        study.stop()


class StudyManager:
    def __init__(self, config: CLConfig) -> None:
        self.config = config
        self.study = self._get_optuna_study()

    def _get_optuna_study(self) -> Study:
        return (
            create_optuna_study(self.config)
            if self.config.use_params_from is None
            else load_optuna_study(self.config)
        )

    @property
    def optuna_study_id(self) -> int:
        return self.study._study_id

    def model_selection(
        self, train_objective: TrainObjective
    ) -> tuple[TrialResult, list[FrozenTrial]]:
        callbacks = [
            MaxTrialsCallback(n_trials=self.config.num_trials),
        ]
        if self.config.skip_model_selection:
            callbacks.append(OneShotCallback())

        self.study.optimize(
            train_objective,  # type: ignore
            timeout=self.config.study_timeout,
            callbacks=callbacks,
        )

        return TrialResult(
            _best_avg_train_loss=self.study.best_trial.user_attrs[
                "best_avg_train_loss"
            ],
            best_avg_train_loss_epoch=self.study.best_trial.user_attrs[
                "best_avg_train_loss_epoch"
            ],
            best_avg_val_acc=self.study.best_trial.user_attrs[
                "best_avg_val_acc"
            ],
            best_avg_val_acc_epoch=self.study.best_trial.user_attrs[
                "best_avg_val_acc_epoch"
            ],
            _best_avg_val_loss=self.study.best_trial.user_attrs[
                "best_avg_val_loss"
            ],
            best_avg_val_loss_epoch=self.study.best_trial.user_attrs[
                "best_avg_val_loss_epoch"
            ],
            trial_number=self.study.best_trial.number,
            params=self.study.best_trial.params,
            val_accs=self.study.best_trial.user_attrs["val_accs"],
            val_losses=self.study.best_trial.user_attrs["val_losses"],
            train_losses_epoch=self.study.best_trial.user_attrs[
                "train_losses_epoch"
            ],
        ), self.study.trials

    def model_assessment(self, test_objective: TestObjective) -> TestResult:
        return test_objective(self.study.best_trial)
