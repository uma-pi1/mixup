import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BaseEnum(StrEnum):
    @classmethod
    def get_member_names(cls) -> list[Any]:
        return list(cls.__members__.keys())


@dataclass
class TestResult:
    _final_train_loss: float | int | None
    test_acc: float
    _test_loss: float
    mixup_graph_ids: list[int] | None

    @property
    def final_train_loss(self) -> float | None:
        return (
            self._final_train_loss
            if not math.isnan(self._final_train_loss)  # type: ignore
            else None
        )

    @property
    def test_loss(self) -> float | None:
        return self._test_loss if not math.isnan(self._test_loss) else None


@dataclass
class TrialResult:
    # Training metrics.
    _best_avg_train_loss: float
    best_avg_train_loss_epoch: float
    best_avg_val_acc: float
    best_avg_val_acc_epoch: int
    _best_avg_val_loss: float
    best_avg_val_loss_epoch: int
    val_accs: list[dict[str, float]]
    val_losses: list[dict[str, float]]
    train_losses_epoch: list[dict[str, float]]
    # Optuna params.
    trial_number: int
    params: dict[str, Any]

    @property
    def best_avg_train_loss(self) -> float | None:
        return (
            self._best_avg_train_loss
            if not math.isnan(self._best_avg_train_loss)
            else None
        )

    @property
    def best_avg_val_loss(self) -> float | None:
        return (
            self._best_avg_val_loss
            if not math.isnan(self._best_avg_val_loss)
            else None
        )
