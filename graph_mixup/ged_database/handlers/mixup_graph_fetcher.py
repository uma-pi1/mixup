import json
import logging
from dataclasses import asdict
from typing import Callable

from sqlalchemy import Select, and_, func, select, union
from sqlalchemy.orm import Session, aliased, selectinload
from sqlalchemy.sql.selectable import ExecutableReturnsRows

from graph_mixup.mixup_generation.typing import (
    MixupDatasetMethodConfig,
)
from graph_mixup.augmentations.typing import GEDFilterFlags
from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.models import GED, Dataset, Edge, Graph, MixupAttr

logger = logging.getLogger(__name__)


graphs_0 = aliased(Graph, name="graphs_0")
graphs_1 = aliased(Graph, name="graphs_1")


class MixupGraphFetcher(BaseHandler):
    def __init__(
        self,
        dataset_name: DatasetName,
        method_name: PreBatchMixupName,
        method_config: MixupDatasetMethodConfig,
        parent_graph_ids: list[int],
        ged_filter_flags: GEDFilterFlags,
        fetch_parents: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.method_config = method_config
        self.parent_graph_ids = parent_graph_ids
        self._additional_filters: list[Callable[[Select], Select]] = (
            _resolve_additional_filters(ged_filter_flags, dataset_name)
        )
        self.fetch_parents = fetch_parents

    def _base_query_builder(self) -> tuple[Select, Select]:
        stmt0 = (
            select(Graph)
            .join(Dataset, Graph.dataset_id == Dataset.dataset_id)
            .join(MixupAttr, MixupAttr.graph_id == Graph.graph_id)
            .outerjoin(
                GED,
                and_(
                    GED.graph_0_id == MixupAttr.parent_0_id,
                    GED.graph_1_id == MixupAttr.parent_1_id,
                ),
            )
            .join(graphs_0, MixupAttr.parent_0_id == graphs_0.graph_id)
            .join(graphs_1, MixupAttr.parent_1_id == graphs_1.graph_id)
        )

        stmt1 = (
            select(Graph)
            .join(Dataset, Graph.dataset_id == Dataset.dataset_id)
            .join(MixupAttr, MixupAttr.graph_id == Graph.graph_id)
            .outerjoin(
                GED,
                and_(
                    GED.graph_0_id == MixupAttr.parent_1_id,
                    GED.graph_1_id == MixupAttr.parent_0_id,
                ),
            )
            .join(graphs_0, MixupAttr.parent_0_id == graphs_0.graph_id)
            .join(graphs_1, MixupAttr.parent_1_id == graphs_1.graph_id)
        )

        stmt0 = self._base_filter(stmt0)
        stmt1 = self._base_filter(stmt1)

        return stmt0, stmt1

    def _base_filter(self, s: Select) -> Select:
        # noinspection PyTypeChecker
        method_config_dict = asdict(self.method_config)

        return (
            s.where(Dataset.name == str(self.dataset_name))
            .where(MixupAttr.mixup_method == str(self.method_name).lower())
            .where(MixupAttr.parent_0_id.in_(self.parent_graph_ids))
            .where(MixupAttr.parent_1_id.in_(self.parent_graph_ids))
            .where(
                func.json_contains(
                    MixupAttr.mixup_hyperparameters,
                    json.dumps(method_config_dict),
                )
            )
        )

    def fetch_mixup_graphs(self) -> list[Graph]:
        stmt0, stmt1 = self._base_query_builder()

        # ===
        # BEGIN: Additional filters.
        # ===

        for f in self._additional_filters:
            stmt0 = f(stmt0)
            stmt1 = f(stmt1)

        # ===
        # END: Additional filters.
        # ===

        stmt = _add_query_options(stmt0, stmt1, self.fetch_parents)

        with Session(self._engine) as session:
            return list(session.execute(stmt).scalars().all())


def _add_query_options(
    stmt0: Select, stmt1: Select, fetch_parents: bool
) -> ExecutableReturnsRows:
    stmt = (
        select(Graph)
        .from_statement(union(stmt0, stmt1))
        .options(
            selectinload(Graph.nodes),
            selectinload(Graph.edges).joinedload(Edge.node_0, innerjoin=True),
            selectinload(Graph.edges).joinedload(Edge.node_1, innerjoin=True),
            selectinload(Graph.mixup_attrs),
        )
    )

    if fetch_parents:
        stmt = stmt.options(
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_0)
            .selectinload(Graph.nodes),
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_0)
            .selectinload(Graph.edges)
            .joinedload(Edge.node_0, innerjoin=True),
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_0)
            .selectinload(Graph.edges)
            .joinedload(Edge.node_1, innerjoin=True),
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_1)
            .selectinload(Graph.nodes),
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_1)
            .selectinload(Graph.edges)
            .joinedload(Edge.node_0, innerjoin=True),
            selectinload(Graph.mixup_attrs)
            .selectinload(MixupAttr.parent_1)
            .selectinload(Graph.edges)
            .joinedload(Edge.node_1, innerjoin=True),
        )

    return stmt


def _resolve_additional_filters(
    flags: GEDFilterFlags,
    dataset_name: DatasetName,
) -> list[Callable[[Select], Select]]:
    filters: list[Callable[[Select], Select]] = []

    # ===
    # GED value.
    # ===

    if flags.max_ged_value is not None:
        logger.info(f"Filter based on GED value: {flags.max_ged_value}")
        filters.append(lambda s: _ged_value_filter(s, flags.max_ged_value))

    # ===
    # Parents' classes.
    # ===

    if flags.only_same_class:
        logger.info("Filter based on parents' classes: same class only")
        filters.append(lambda s: _parents_class_filter(s, True))
    if flags.only_different_class:
        logger.info("Filter based on parents' classes: different class only")
        filters.append(lambda s: _parents_class_filter(s, False))

    # ===
    # Absolute GED distribution.
    # ===

    if flags.only_first_absolute_quintile:
        logger.info("Filter based on absolute GED distribution: first quintile")
        filters.append(
            lambda s: _absolute_ged_ntile_filter(s, 5, 1, dataset_name)
        )
    if flags.only_last_absolute_quintile:
        logger.info("Filter based on absolute GED distribution: last quintile")
        filters.append(
            lambda s: _absolute_ged_ntile_filter(s, 5, 5, dataset_name)
        )

    # ===
    # Relative GED distribution.
    # ===

    if flags.only_first_relative_quintile:
        logger.info("Filter based on relative GED distribution: first quintile")
        filters.append(
            lambda s: _relative_ged_ntile_filter(s, 5, 1, dataset_name)
        )
    if flags.only_last_relative_quintile:
        logger.info("Filter based on relative GED distribution: last quintile")
        filters.append(
            lambda s: _relative_ged_ntile_filter(s, 5, 5, dataset_name)
        )

    return filters


def _ged_value_filter(s: Select, max_value: int) -> Select:
    return s.where(GED.value <= max_value)


def _parents_class_filter(s: Select, from_same_class: bool) -> Select:
    # noinspection PyTypeChecker
    return (
        s.where(graphs_0.label == graphs_1.label)
        if from_same_class
        else s.where(graphs_0.label != graphs_1.label)
    )


def _absolute_ged_ntile_filter(
    s: Select, n_tiles: int, n_tile: int, dataset_name: DatasetName
) -> Select:
    # noinspection PyTypeChecker
    n_tile_subquery = (
        select(
            GED.ged_id,
            func.ntile(n_tiles).over(order_by=GED.value).label("n_tile"),
        )
        .join(graphs_0, GED.graph_0_id == graphs_0.graph_id)
        .join(graphs_1, GED.graph_1_id == graphs_1.graph_id)
        # ===
        # Filter dataset.
        # ===
        .join(Dataset, Dataset.dataset_id == graphs_0.dataset_id)
        .where(graphs_0.dataset_id == graphs_1.dataset_id)
        .where(Dataset.name == dataset_name)
        # ===
        # Exclude mixup graphs.
        # ===
        .where(
            ~(select(1).where(MixupAttr.graph_id == graphs_0.graph_id).exists())
        )
        .where(
            ~(select(1).where(MixupAttr.graph_id == graphs_1.graph_id).exists())
        )
        # ===
        # Exclude unsuccessful GEDs.
        # ===
        .where(GED.value > -1)
        .subquery()
    )
    return s.join(
        n_tile_subquery, GED.ged_id == n_tile_subquery.c.ged_id
    ).where(n_tile == n_tile_subquery.c.n_tile)


def _relative_ged_ntile_filter(
    s: Select, n_tiles: int, n_tile: int, dataset_name: DatasetName
) -> Select:
    # noinspection PyTypeChecker
    n_tile_subquery = (
        select(
            GED.ged_id,
            func.ntile(n_tiles)
            .over(partition_by=graphs_0.graph_id, order_by=GED.value)
            .label("n_tile"),
        )
        .join(graphs_0, GED.graph_0_id == graphs_0.graph_id)
        .join(graphs_1, GED.graph_1_id == graphs_1.graph_id)
        # ===
        # Filter dataset.
        # ===
        .join(Dataset, Dataset.dataset_id == graphs_0.dataset_id)
        .where(graphs_0.dataset_id == graphs_1.dataset_id)
        .where(Dataset.name == dataset_name)
        # ===
        # Exclude mixup graphs.
        # ===
        .where(
            ~(select(1).where(MixupAttr.graph_id == graphs_0.graph_id).exists())
        )
        .where(
            ~(select(1).where(MixupAttr.graph_id == graphs_1.graph_id).exists())
        )
        # ===
        # Exclude unsuccessful GEDs.
        # ===
        .where(GED.value > -1)
        .subquery()
    )
    return s.join(
        n_tile_subquery, GED.ged_id == n_tile_subquery.c.ged_id
    ).where(n_tile == n_tile_subquery.c.n_tile)
