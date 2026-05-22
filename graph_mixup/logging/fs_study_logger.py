import os
from dataclasses import asdict

import yaml
from optuna.trial import FrozenTrial
from typing_extensions import override

from graph_mixup.config.typing import CLConfig
from graph_mixup.logging.base_logger import BaseLogger
from graph_mixup.logging.utils import value_mean_std_dict
from graph_mixup.resource_locators import ResourceLocator
from graph_mixup.typing import TrialResult, TestResult


class FileSystemStudyLogger(BaseLogger):
    def __init__(
        self, config: CLConfig, commit_hash: str, optuna_study_id: int
    ) -> None:
        self.locator = ResourceLocator(config)
        self.test_results: list[TestResult] = []

        super().__init__(config, commit_hash, optuna_study_id)

    @override
    def _create_or_fetch_experiment(
        self, config: CLConfig, commit_hash: str
    ) -> int:
        os.makedirs(self.locator.get_root_log_dir(), exist_ok=True)
        os.makedirs(self.locator.get_experiment_log_dir_path(), exist_ok=True)

        with open(self.locator.get_config_dump_path(), "w") as f:
            yaml.dump(asdict(config), f)

        return 0

    @override
    def _create_study(self, optuna_study_id: int, fold: int) -> int:
        return 0

    @override
    def log_model_selection_result(
        self, best_trial_result: TrialResult, _: list[FrozenTrial]
    ) -> None:
        with open(self.locator.get_best_trial_file_path(), "a") as f:
            yaml.dump(
                dict(
                    model_selection=dict(
                        best_avg_train_loss=best_trial_result.best_avg_train_loss,
                        best_avg_train_loss_epoch=best_trial_result.best_avg_train_loss_epoch,
                        best_avg_val_acc=best_trial_result.best_avg_val_acc,
                        best_avg_val_acc_epoch=best_trial_result.best_avg_val_acc_epoch,
                        best_avg_val_loss=best_trial_result.best_avg_val_loss,
                        best_avg_val_loss_epoch=best_trial_result.best_avg_val_loss_epoch,
                        trial_number=best_trial_result.trial_number,
                        params=best_trial_result.params,
                    )
                ),
                f,
            )

    @override
    def log_model_assessment_result(self, test_result: TestResult) -> None:
        self.test_results.append(test_result)

    @override
    def log_study_completed(self) -> None:
        with open(self.locator.get_best_trial_file_path(), "a") as f:
            yaml.dump(
                dict(
                    model_assessment=dict(
                        test_accs=value_mean_std_dict(
                            [tr.test_acc for tr in self.test_results]
                        ),
                        test_losses=value_mean_std_dict(
                            [tr.test_loss for tr in self.test_results]
                        ),
                        final_train_losses=value_mean_std_dict(
                            [tr.final_train_loss for tr in self.test_results]
                        ),
                    ),
                ),
                f,
            )

        with open(
            os.path.join(
                self.locator.get_experiment_log_dir_path(), "_SUCCESS"
            ),
            "w",
        ) as f:
            f.write("")

    @override
    def log_study_incomplete(self, e: Exception) -> None:
        with open(
            os.path.join(self.locator.get_experiment_log_dir_path(), "_ERROR"),
            "w",
        ) as f:
            f.write(str(e))

    @override
    def dispose(self) -> None:
        return
