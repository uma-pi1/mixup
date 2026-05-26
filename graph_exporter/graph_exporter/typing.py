from dataclasses import dataclass
from typing import Any, Literal


# ===
# Method Configs.
# ===


@dataclass
class BaseConfig:
    seed: int


# ===
# GeoMix.
# ===


@dataclass
class GeoMixConfig(BaseConfig):
    num_graphs: int  # default: 10
    num_nodes: int  # default: 20 for IMDB/MUTAG & 40 for PROTEINS/MSRC_9
    alpha_fgw: float  # default: 1.0
    sample_dist: Literal["uniform", "beta"]  # default: uniform
    uniform_min: float  # default: 0.0
    uniform_max: float  # default: 5e-2
    clip_eps: float  # default: 1e-3
    mixup_alpha: float  # default (5.0, 0.5)


# ===
# FGWMixup.
# ===


@dataclass
class FGWMixupConfig(BaseConfig):
    measure: Literal["degree", "uniform"]  # default: degree
    metric: Literal["sp", "adj"]  # default: sp
    fgw_alpha: float  # default: 0.95
    loss_fun: Literal["square_loss", "kl_loss"]  # default: square_loss
    mixup_alpha: float  # default: 0.2 (called "beta_k")
    rho: float  # hp search over {0.1, 1, 10}


# ===
# IfMixup.
# ===


@dataclass
class IfMixupConfig(BaseConfig):
    mixup_alpha: float  # default: Beta(20, 1)


# ===
# SubMix.
# ===


@dataclass
class SubMixConfig(BaseConfig):
    aug_size: float


# ===
# Import / Export.
# ===


@dataclass
class MixupItem:
    graph_dict: dict[str, Any]
    lam: float
    source_indices: tuple[int, int]
    creation_time_us: int


@dataclass
class ImportData:
    mixup_items: list[MixupItem]
    config: BaseConfig
