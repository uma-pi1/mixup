import logging
from multiprocessing import Pool

from tqdm import tqdm

from graph_mixup.compute_ged.computers.exact_computer import ExactGEDComputer
from graph_mixup.compute_ged.computers.gedlib_computer import GEDLibComputer
from graph_mixup.compute_ged.parser import Args, parse_args
from graph_mixup.compute_ged.typing import GEDResult
from graph_mixup.ged_database.handlers.ged_compute_database_handler import (
    GEDComputeDatabaseHandler,
)


class GEDComputeManager:
    def __init__(self, args: Args) -> None:
        self.db_manager = GEDComputeDatabaseHandler()

        # Mixup Parameters.
        self.dataset_name = args.dataset_name
        self.method_name = args.method_name

        # Computation Parameters.
        self.n_cpus = args.n_cpus
        self.timeout = args.timeout
        self.lb_threshold = args.lb_threshold
        self.batch_size = args.batch_size
        self.exec = args.exec
        self.ged_approx_method = args.approx_method

    def compute_geds_and_store(self) -> None:
        if self.method_name is None:
            batches = self.db_manager.get_graph_pairs_without_ged(
                self.dataset_name, limit=self.batch_size
            )
        else:
            batches = self.db_manager.get_mixup_graph_pairs_without_ged(
                self.dataset_name, self.method_name, limit=self.batch_size
            )

        computer = (
            ExactGEDComputer(self.timeout, self.lb_threshold)
            if self.ged_approx_method is None
            else GEDLibComputer(
                self.timeout,
                self.lb_threshold,
                self.exec,
                self.ged_approx_method,
            )
        )

        results: list[GEDResult] = []
        for batch in tqdm(batches):
            with Pool(self.n_cpus) as p:
                results += p.starmap(
                    computer.process,
                    batch,
                )

            for result in results:
                self.db_manager.create_ged_result(result)
            results = []


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Args: {args}")

    manager = GEDComputeManager(args)
    manager.compute_geds_and_store()
