from typing import Type

from optuna.trial import FrozenTrial
from typing_extensions import override

from graph_mixup.config.typing import CLConfig
from graph_mixup.logging.base_logger import BaseLogger, ILogger
from graph_mixup.typing import TrialResult, TestResult


class LoggerContainer(ILogger):
    def __init__(
        self,
        config: CLConfig,
        commit_hash: str,
        optuna_study_id: int,
        *loggers: Type[BaseLogger],
    ) -> None:
        self.loggers = [
            Logger(config, commit_hash, optuna_study_id) for Logger in loggers
        ]

    @override
    def log_model_selection_result(
        self, best_trial_result: TrialResult, all_trials: list[FrozenTrial]
    ) -> None:
        for logger in self.loggers:
            logger.log_model_selection_result(best_trial_result, all_trials)

    @override
    def log_model_assessment_result(self, test_result: TestResult) -> None:
        for logger in self.loggers:
            logger.log_model_assessment_result(test_result)

    @override
    def log_study_incomplete(self, e: Exception) -> None:
        for logger in self.loggers:
            logger.log_study_incomplete(e)

    @override
    def log_study_completed(self) -> None:
        for logger in self.loggers:
            logger.log_study_completed()

    @override
    def dispose(self) -> None:
        for logger in self.loggers:
            logger.dispose()
