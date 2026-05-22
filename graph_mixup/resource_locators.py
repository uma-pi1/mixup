import os
from abc import ABC

from graph_mixup.config.typing import CLConfig


class AbstractResourceLocator(ABC):
    """
    Provides paths and names for files, directories, and other resources (e.g.,
    optuna study name). Does not create directories or files.
    """

    CHKPT_DIRNAME: str = "checkpoints"

    def __init__(self, config: CLConfig, *, from_baseline: bool) -> None:
        self.config = config
        self.from_baseline = from_baseline

    def get_root_log_dir(self) -> str:
        return self.config.log_dir

    def get_fold_name(self) -> str:
        return f"fold_{self.config.fold}"

    def get_trial_name(self, trial_number: int) -> str:
        return f"trial_{trial_number}"

    def get_test_round_name(self, test_round_number: int) -> str:
        return f"test_round_{test_round_number}"

    def get_optuna_study_name(self) -> str:
        return self._get_experiment_identifier() + "_" + self.get_fold_name()

    def get_config_dump_path(self) -> str:
        return os.path.join(
            self.get_experiment_log_dir_path(),
            "config.yml",
        )

    def get_experiment_log_dir_path(self) -> str:
        return os.path.join(
            self.get_root_log_dir(),
            self._get_experiment_identifier(),
        )

    def get_best_trial_file_path(self) -> str:
        best_trial_path = os.path.join(
            self.get_experiment_log_dir_path(),
            self.get_fold_name() + "_best_trial.yml",
        )

        return best_trial_path

    def get_trial_checkpoint_path(self, trial_number: int) -> str:
        return os.path.join(
            self._get_study_dir(),
            self.get_trial_name(trial_number),
            self.CHKPT_DIRNAME,
        )

    def get_eval_checkpoint_path(self, test_round: int) -> str:
        return os.path.join(
            self._get_study_dir(),
            self.get_test_round_name(test_round),
            self.CHKPT_DIRNAME,
        )

    # ===
    # Private Methods
    # ===

    def _get_study_dir(self) -> str:
        return os.path.join(
            self.get_experiment_log_dir_path(), self.get_fold_name()
        )

    def _get_experiment_identifier(self) -> str:
        method_name = "None" if self.from_baseline else self.config.method_name
        use_baseline = (
            "False" if self.from_baseline else self.config.use_baseline
        )

        return (
            f"{self.config.model_name}_{self.config.dataset_name}_"
            f"{method_name}_"
            f"num_trials_{self.config.num_trials}_"
            f"study_timeout_{self.config.study_timeout}_"
            f"train_timeout_{self.config.train_timeout}_"
            f"seed_{self.config.seed}_cv_seed_{self.config.cv_seed}_"
            f"num_outer_folds_{self.config.num_outer_folds}_"
            f"num_inner_folds_{self.config.num_inner_folds}_"
            f"use_inner_holdout_{self.config.use_inner_holdout}_"
            f"num_test_rounds_{self.config.num_test_rounds}_"
            f"max_epochs_{self.config.max_epochs}_"
            f"use_baseline_{use_baseline}_"
            f"using_fixed_params_{self.config.use_params_from is not None}"
        )


class ResourceLocator(AbstractResourceLocator):
    def __init__(self, config: CLConfig):
        super().__init__(config, from_baseline=False)


class BaselineResourceLocator(AbstractResourceLocator):
    def __init__(self, config: CLConfig):
        super().__init__(config, from_baseline=True)
