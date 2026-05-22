import logging

from graph_mixup.mixup_generation.submix.mixup_producer import SubMixProducer
from graph_mixup.mixup_generation.submix.parser import SubMixGenerationParser


def main() -> None:
    parser = SubMixGenerationParser()
    args = parser.parse_args()

    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARN))

    generator = SubMixProducer(
        args.batch_size,
        args.seed,
        args.dataset_name,
        args.max_items_per_pair,
        args.aug_size,
    )

    generator.produce(args.max_total)


if __name__ == "__main__":
    main()
