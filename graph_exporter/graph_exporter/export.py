import os
import pickle
import sys
import time
from dataclasses import asdict

import yaml

from graph_exporter.typing import MixupItem, BaseConfig


def export(
    mixup_items: list[MixupItem],
    dataset_name: str,
    config: BaseConfig,
    *,
    base_dir: str = "export",
    terminate: bool = True,
) -> None:
    path = os.path.join(base_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    with open(os.path.join(path, timestamp + "_graphs.pkl"), "wb") as f:
        pickle.dump((config, mixup_items), f)

    with open(os.path.join(path, timestamp + "_config.yml"), "w") as f:
        yaml.dump(asdict(config), f)

    if terminate:
        sys.exit(0)
