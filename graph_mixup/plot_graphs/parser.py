from argparse import ArgumentParser
from dataclasses import dataclass

from graph_mixup.config.typing import DatasetName, PreBatchMixupName


@dataclass
class Args:
    dataset_name: DatasetName
    method_name: PreBatchMixupName
    out_dir: str
    seed: int


def parse() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--method_name", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    return Args(**vars(parser.parse_args()))
