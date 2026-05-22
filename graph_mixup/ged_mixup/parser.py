from dataclasses import dataclass
from typing_extensions import override

from graph_mixup.mixup_generation.parser import (
    BaseArgs,
    BaseMixupGenerationParser,
)


@dataclass
class GEDMixupArgs(BaseArgs):
    lam: float | None
    mixup_alpha: float | None
    max_fail_count: int


class GEDMixupGenerationParser(BaseMixupGenerationParser):
    @override
    def _add_custom_args(self) -> None:
        # Lambda xor mixup alpha.
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--lam", type=float, help="Mixup Lambda")
        group.add_argument(
            "--mixup_alpha", type=float, help="Parameter for Beta distribution"
        )

        # Max fail count before advancing to next graph pair.
        self.parser.add_argument(
            "--max_fail_count", "-mfc", type=int, default=5
        )

    @override
    def parse_args(self) -> GEDMixupArgs:
        parsed_args = self.parser.parse_args()
        return GEDMixupArgs(**vars(parsed_args))
