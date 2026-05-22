from dataclasses import dataclass

from graph_exporter.typing import BaseConfig


@dataclass
class GMixupConfig(BaseConfig):
    mixup_alpha: float
