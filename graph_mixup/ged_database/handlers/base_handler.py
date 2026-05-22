from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.models import (
    Dataset,
    Graph,
    MixupAttr,
    Edge,
)
from graph_mixup.ged_database.utils import get_engine


class DatasetNotFound(Exception): ...


class BaseHandler:
    def __init__(self) -> None:
        self._engine = get_engine()

    def get_dataset(self, dataset_name: DatasetName) -> Dataset:
        with Session(self._engine) as session:
            return self._get_dataset(session, dataset_name)

    def get_vanilla_graphs(self, dataset_name: DatasetName) -> list[Graph]:
        stmt = (
            select(Graph)
            .join(Dataset, Dataset.dataset_id == Graph.dataset_id)
            .outerjoin(MixupAttr, MixupAttr.graph_id == Graph.graph_id)
            .where(Dataset.name == dataset_name)
            .where(MixupAttr.mixup_attr_id.is_(None))
            .options(
                selectinload(Graph.nodes),
                selectinload(Graph.edges).joinedload(
                    Edge.node_0, innerjoin=True
                ),
                selectinload(Graph.edges).joinedload(
                    Edge.node_1, innerjoin=True
                ),
            )
        )

        with Session(self._engine) as session:
            graphs = session.execute(stmt).scalars().all()

        return list(graphs)

    @staticmethod
    def _get_dataset(session: Session, dataset_name: DatasetName) -> Dataset:
        dataset = session.scalars(
            select(Dataset).where(Dataset.name == dataset_name)
        ).first()

        if dataset is None:
            raise DatasetNotFound()

        return dataset
