from argparse import ArgumentParser
from dataclasses import dataclass

from graph_mixup.config.typing import DatasetName


@dataclass
class BaseArgs:
    dataset_name: DatasetName
    batch_size: int
    max_items_per_pair: int
    max_total: int
    seed: int
    verbose: bool


class BaseMixupGenerationParser:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self._add_default_args()
        self._add_custom_args()

    def parse_args(self) -> BaseArgs:
        """Overwrite this method to parse custom arguments."""
        parsed_args = self.parser.parse_args()
        return BaseArgs(**vars(parsed_args))

    def _add_custom_args(self) -> None:
        """Override this method to add custom arguments."""
        pass

    def _add_default_args(self) -> None:
        self.parser.add_argument(
            "--dataset_name", type=DatasetName, required=True
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
        )
        self.parser.add_argument(
            "--max_items_per_pair",
            type=int,
            default=1,
            help="Maximum number of items per GED mapping. Each item uses a "
            "different GED path.",
        )
        self.parser.add_argument(
            "--max_total",
            type=int,
            required=True,
            help="Maximum number of mixup items to generate.",
        )
        self.parser.add_argument("--seed", type=int, default=0)
        self.parser.add_argument("--verbose", "-v", action="store_true")
