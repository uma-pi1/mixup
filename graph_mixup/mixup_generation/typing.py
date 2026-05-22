from dataclasses import dataclass

from graph_mixup.augmentations.data.typing import (
    AbstractDatasetMethodConfig,
)


@dataclass
class MixupDatasetMethodConfig(AbstractDatasetMethodConfig):
    """Base class for all mixup method configs. Can also be used directly if
    a method does not require custom parameters."""

    mixup_alpha: float
