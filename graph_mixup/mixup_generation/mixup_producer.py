from abc import abstractmethod, ABC
from typing import Generator

from tqdm import trange

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.models import Graph


class MixupProducer(ABC):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        dataset_name: DatasetName,
        max_items_per_pair: int,
    ) -> None:
        # Environment Params
        self.batch_size = batch_size
        self.seed = seed
        # Mixup Params
        self.dataset_name = dataset_name
        self.max_items_per_pair = max_items_per_pair

    def produce(self, max_total: int) -> list[Graph]:
        graphs: list[Graph] = []

        generator = self.produce_generator()
        for _ in trange(max_total):
            graphs.append(next(generator))
        return graphs

    @abstractmethod
    def produce_generator(self) -> Generator[Graph, None, None]: ...
