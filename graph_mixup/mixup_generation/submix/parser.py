from dataclasses import dataclass
from typing_extensions import override

from graph_mixup.mixup_generation.parser import (
    BaseArgs,
    BaseMixupGenerationParser,
)


@dataclass
class SubMixArgs(BaseArgs):
    aug_size: float


class SubMixGenerationParser(BaseMixupGenerationParser):
    @override
    def _add_custom_args(self) -> None:
        # Lambda xor mixup alpha.
        self.parser.add_argument("--aug_size", type=float, default=0.4)

    @override
    def parse_args(self) -> SubMixArgs:
        parsed_args = self.parser.parse_args()
        return SubMixArgs(**vars(parsed_args))
