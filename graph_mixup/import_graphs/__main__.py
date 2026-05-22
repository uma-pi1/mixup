import logging

from graph_exporter.load import load
from graph_exporter.typing import ImportData
from tqdm import tqdm

from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.graph_import_database_handler import (
    GraphImportDatabaseHandler,
)
from graph_mixup.import_graphs.parser import parse


def store_dataset(dataset_name: DatasetName) -> None:
    db_manager = GraphImportDatabaseHandler()
    db_manager.get_or_create_dataset(dataset_name, "data")


def store_data(
    data: list[ImportData],
    dataset_name: DatasetName,
    method_name: PreBatchMixupName,
    sample_edges: bool,
) -> None:
    db_manager = GraphImportDatabaseHandler()
    for item in tqdm(data):
        db_manager.create_mixup_graphs(
            item.mixup_items,
            item.config,
            dataset_name,
            method_name,
            sample_edges,
        )


def main() -> None:
    args = parse()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARN)

    if args.method_name is not None:
        if args.path is None:
            raise ValueError("Path to mixup graphs' root dir must be provided.")
        print("Load data ... ")
        data = load(args.path)
        print("Data loaded.")
        print("Store data ... ")
        store_data(data, args.dataset_name, args.method_name, args.sample_edges)
        print("Data stored.")
    else:
        store_dataset(args.dataset_name)


if __name__ == "__main__":
    main()
