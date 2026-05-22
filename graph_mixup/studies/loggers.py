from argparse import Namespace
from collections import defaultdict
from typing import Union, Any, Optional

from pytorch_lightning.loggers.logger import Logger
from typing_extensions import override


class InMemoryLogger(Logger):
    """Stores metrics in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.stepwise_metrics: dict[int, dict[str, float]] = defaultdict(
            dict
        )  # Store metrics as {step: {metric_name: value}}

    @override
    @property
    def name(self) -> str:
        return "SimpleLogger"

    @override
    @property
    def version(self) -> str | int:
        return "0.1"

    @override
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None

    @override
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        if step is None:
            raise ValueError("Step is required for this logger.")

        self.stepwise_metrics[step].update(metrics)

    @override
    def finalize(self, status: str) -> None:
        pass

    def get_epochwise_metric(
        self, metric_name: str, every_n_th_epoch: int = 10
    ) -> dict[int, float]:
        epochwise_metric: dict[int, float] = dict()

        for step, step_metrics in self.stepwise_metrics.items():
            if metric_name in step_metrics:
                if "epoch" not in step_metrics:
                    raise ValueError(
                        f"epoch must be logged alongside {metric_name} (default"
                        f" behavior with `on_epoch=True`)"
                    )

                epoch = int(step_metrics["epoch"])

                if (epoch + 1) % every_n_th_epoch == 0:
                    epochwise_metric[epoch] = step_metrics[metric_name]

        return epochwise_metric

    def get_metric_values(self, metric_name: str) -> list[float]:
        metric: list[float] = []

        for step, step_metrics in self.stepwise_metrics.items():
            if metric_name in step_metrics:
                metric.append(step_metrics[metric_name])

        return metric

    def get_metric_value(self, metric_name: str) -> float:
        metric = self.get_metric_values(metric_name)
        if len(metric) != 1:
            raise ValueError(f"Metric {metric_name} must have exactly 1 value.")
        return metric[0]
