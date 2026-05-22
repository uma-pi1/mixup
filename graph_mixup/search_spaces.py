import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from optuna import Trial
from typing_extensions import override


logger = logging.getLogger(__name__)


class SearchSpace(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def suggest(self, trial: Trial):
        """Suggest a concrete hyperparameter value."""


T = TypeVar("T")


class CategoricalSearchSpace(SearchSpace, Generic[T]):
    def __init__(self, name: str, choices: list[T]):
        super().__init__(name)
        self.choices = choices

    @override
    def suggest(self, trial: Trial) -> T:
        val = trial.suggest_categorical(self.name, self.choices)
        logger.info(
            f"Suggesting categorical {self.name} from {self.choices}: {val}"
        )
        return val  # type: ignore


class FloatSearchSpace(SearchSpace):
    def __init__(
        self, name: str, low: float, high: float, *, log: bool = False
    ):
        super().__init__(name)
        self.low = low
        self.high = high
        self.log = log

    @override
    def suggest(self, trial: Trial) -> float:
        val = trial.suggest_float(self.name, self.low, self.high, log=self.log)
        logger.info(
            f"Suggesting float {self.name} from [{self.low}, {self.high}] (log={self.log}): {val}"
        )
        return val


class IntSearchSpace(SearchSpace):
    def __init__(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ):
        super().__init__(name)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    @override
    def suggest(self, trial: Trial) -> int:
        val = trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )
        logger.info(
            f"Suggesting int {self.name} from {self.low}, ..., {self.high} (step={self.step}, log={self.log}): {val}"
        )
        return val
