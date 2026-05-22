from abc import ABC, abstractmethod

from optuna.trial import FrozenTrial

from graph_mixup.config.typing import CLConfig
from graph_mixup.typing import TrialResult, TestResult


class ILogger(ABC):
    """
    Experiment: Consists of multiple studies.
    Study: Contains results for a single fold. Belongs to a single experiment.
    """

    @abstractmethod
    def log_model_selection_result(
        self, best_trial_result: TrialResult, all_trials: list[FrozenTrial]
    ) -> None: ...

    @abstractmethod
    def log_model_assessment_result(self, test_result: TestResult) -> None: ...

    @abstractmethod
    def log_study_completed(self) -> None: ...

    @abstractmethod
    def log_study_incomplete(self, e: Exception) -> None: ...

    @abstractmethod
    def dispose(self) -> None: ...


class BaseLogger(ILogger, ABC):
    def __init__(
        self, config: CLConfig, commit_hash: str, optuna_study_id: int
    ) -> None:
        self.optuna_study_id = optuna_study_id

        self.experiment_id = self._create_or_fetch_experiment(
            config, commit_hash
        )
        self.study_id = self._create_study(optuna_study_id, config.fold)

    @abstractmethod
    def _create_or_fetch_experiment(
        self, config: CLConfig, commit_hash: str
    ) -> int: ...

    @abstractmethod
    def _create_study(self, optuna_study_id: int, fold: int) -> int: ...
