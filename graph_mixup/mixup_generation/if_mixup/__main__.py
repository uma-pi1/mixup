import logging

from graph_mixup.mixup_generation.if_mixup.mixup_producer import IfMixupProducer
from graph_mixup.mixup_generation.if_mixup.parser import IfMixupGenerationParser


def main() -> None:
    parser = IfMixupGenerationParser()
    args = parser.parse_args()

    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARN))

    generator = IfMixupProducer(
        args.batch_size,
        args.seed,
        args.dataset_name,
        args.lam,
        args.mixup_alpha,
        args.max_items_per_pair,
        args.sample_edges,
    )

    generator.produce(args.max_total)


if __name__ == "__main__":
    main()
