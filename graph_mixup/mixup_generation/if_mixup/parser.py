from dataclasses import dataclass
from typing_extensions import override

from graph_mixup.mixup_generation.parser import (
    BaseArgs,
    BaseMixupGenerationParser,
)


@dataclass
class IfMixupArgs(BaseArgs):
    lam: float | None
    mixup_alpha: float | None
    sample_edges: bool


class IfMixupGenerationParser(BaseMixupGenerationParser):
    @override
    def _add_custom_args(self) -> None:
        # Lambda xor mixup alpha.
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--lam", type=float, help="Mixup Lambda")
        group.add_argument(
            "--mixup_alpha", type=float, help="Parameter for Beta distribution"
        )
        # Sample edges.
        self.parser.add_argument(
            "--sample_edges",
            "-se",
            default=False,
            action="store_true",
            help="Sample edges using Bernoulli distribution with their weights.",
        )

    @override
    def parse_args(self) -> IfMixupArgs:
        parsed_args = self.parser.parse_args()
        return IfMixupArgs(**vars(parsed_args))
