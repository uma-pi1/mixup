from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from typing import cast

from graph_mixup.config.typing import DatasetName, PreBatchMixupName


@dataclass
class Args:
    path: str | None
    dataset_name: DatasetName
    method_name: PreBatchMixupName | None
    sample_edges: bool
    verbose: bool


def parse() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="path to the root directory of the mixup graphs",
    )
    parser.add_argument(
        "--dataset_name",
        type=DatasetName,
        choices=[cast(Enum, dataset).value for dataset in DatasetName],
        required=True,
    )
    parser.add_argument(
        "--method_name",
        type=PreBatchMixupName,
        choices=[
            cast(Enum, method).value for method in list(PreBatchMixupName)
        ],
    )
    parser.add_argument(
        "--sample_edges",
        "-se",
        default=False,
        action="store_true",
        help="Sample edges using Bernoulli distribution with their weights.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
    )
    return Args(**vars(parser.parse_args()))
