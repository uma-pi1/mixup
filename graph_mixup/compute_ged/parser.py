import os
from argparse import ArgumentParser
from dataclasses import dataclass

from graph_mixup.compute_ged.typing import GEDLIBMethod
from graph_mixup.config.typing import DatasetName, PreBatchMixupName


@dataclass
class Args:
    dataset_name: DatasetName
    n_cpus: int
    timeout: int
    lb_threshold: int
    batch_size: int
    method_name: PreBatchMixupName
    verbose: bool
    approx_method: GEDLIBMethod | None
    exec: str


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=DatasetName, required=True)
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=os.cpu_count() - 2,
        help="Number of CPUs to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Timeout in seconds for each GED calculation",
    )
    parser.add_argument(
        "--lb_threshold",
        type=int,
        default=1000,
        help="Lower bound threshold. If the computed lower bound is above this "
        "threshold, GED computation is skipped.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--method_name",
        type=str,
        help="Compute GED of mixup graphs with both their parents, and between "
        "their parents.",
    )
    parser.add_argument(
        "--approx_method",
        type=GEDLIBMethod,
        choices=[m.value for m in GEDLIBMethod],
        help=(
            "If present, the GED will be approximated with the chosen method. "
            "Otherwise, exact GED will be computed."
        ),
    )
    parser.add_argument(
        "--exec",
        default=os.path.join(
            os.path.dirname(__file__), "portable_edit_path", "edit_path_exec"
        ),
        help="Path to the GED executable for edit path extraction.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return Args(**vars(parser.parse_args()))
