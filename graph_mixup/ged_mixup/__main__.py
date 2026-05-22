import logging
import sys

from graph_mixup.ged_mixup.mixup_producer import GEDMixupProducer
from graph_mixup.ged_mixup.parser import GEDMixupGenerationParser


def main() -> None:
    parser = GEDMixupGenerationParser()
    args = parser.parse_args()

    sys.setrecursionlimit(10_000)

    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARN))

    generator = GEDMixupProducer(
        args.batch_size,
        args.seed,
        args.dataset_name,
        args.lam,
        args.mixup_alpha,
        args.max_items_per_pair,
        args.max_fail_count,
    )

    generator.produce(args.max_total)


if __name__ == "__main__":
    main()
