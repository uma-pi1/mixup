from dataclasses import dataclass
from typing import TypeVar

from graph_exporter.typing import BaseConfig

from graph_mixup.augmentations.data.typing import (
    DatasetConfig,
    MethodConfigType,
)


@dataclass
class GEDFilterFlags:
    max_ged_value: int | None
    only_same_class: bool
    only_different_class: bool
    only_first_absolute_quintile: bool
    only_last_absolute_quintile: bool
    only_first_relative_quintile: bool
    only_last_relative_quintile: bool

    def __post_init__(self) -> None:
        if self.only_same_class and self.only_different_class:
            raise ValueError(
                "only_same_class and only_different_class cannot both be true at the same time."
            )
        if (
            self.only_first_absolute_quintile
            and self.only_last_absolute_quintile
        ):
            raise ValueError(
                "only_first_absolute_quintile and only_last_absolute_quintile cannot both be true at the same time."
            )
        if (
            self.only_first_relative_quintile
            and self.only_last_relative_quintile
        ):
            raise ValueError(
                "only_first_relative_quintile and only_last_relative_quintile cannot both be true at the same time."
            )


@dataclass
class AugDatasetConfig(DatasetConfig[MethodConfigType | BaseConfig]):
    """Base class of all mixup dataset configs."""

    use_vanilla: bool
    augmented_ratio: float
    ged_filter_flags: GEDFilterFlags


AugDatasetConfigType = TypeVar("AugDatasetConfigType", bound=AugDatasetConfig)
