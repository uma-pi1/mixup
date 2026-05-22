import logging
from typing import Generator

from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased, selectinload

from graph_mixup.compute_ged.typing import GEDResult
from graph_mixup.config.typing import DatasetName, PreBatchMixupName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.models import (
    GED,
    Dataset,
    Edge,
    Graph,
    Mapping,
    MixupAttr,
)


logger = logging.getLogger(__name__)


class GEDComputeDatabaseHandler(BaseHandler):
    def create_ged_result(self, result: GEDResult) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                ged_sql = GED(
                    graph_0_id=result.graph_0_id,
                    graph_1_id=result.graph_1_id,
                    value=result.value,
                    time=result.time,
                    lower_bound=result.lb,
                )
                session.add(ged_sql)
                session.flush()

                if result.mapping is not None:
                    graph_0_ids = {
                        node.index: node.node_id
                        for node in ged_sql.graph_0.nodes
                    }
                    graph_1_ids = {
                        node.index: node.node_id
                        for node in ged_sql.graph_1.nodes
                    }

                    for graph_0_idx, graph_1_idx in result.mapping.items():
                        session.add(
                            Mapping(
                                ged_id=ged_sql.ged_id,
                                node_0_id=graph_0_ids[graph_0_idx],
                                node_1_id=graph_1_ids[graph_1_idx],
                            )
                        )
        except IntegrityError as e:
            logger.warning("Exception caught (continuing): %s", e)

    def get_mixup_graph_pairs_without_ged(
        self,
        dataset_name: DatasetName,
        method_name: PreBatchMixupName,
        limit: int,
    ) -> Generator[list[tuple[Graph, Graph]], None, None]:
        while True:
            graphs_m = aliased(Graph, name="graphs_m")
            graphs_p = aliased(Graph, name="graphs_p")
            ged_0 = aliased(GED, name="ged_0")
            ged_1 = aliased(GED, name="ged_1")

            first_parent = (
                select(graphs_p, graphs_m)
                .join(Dataset, Dataset.dataset_id == graphs_m.dataset_id)
                .join(MixupAttr, MixupAttr.graph_id == graphs_m.graph_id)
                .join(
                    graphs_p, MixupAttr.parent_0_id == graphs_p.graph_id
                )  # join with first parent
                .outerjoin(
                    ged_0,
                    and_(
                        ged_0.graph_0_id == graphs_p.graph_id,
                        ged_0.graph_1_id == graphs_m.graph_id,
                    ),
                )
                .outerjoin(
                    ged_1,
                    and_(
                        ged_1.graph_0_id == graphs_m.graph_id,
                        ged_1.graph_1_id == graphs_p.graph_id,
                    ),
                )
                .where(Dataset.name == dataset_name)
                .where(MixupAttr.mixup_method == method_name)
                .where(
                    and_(ged_0.value.is_(None), ged_1.value.is_(None)),
                )
                .options(
                    selectinload(graphs_m.nodes),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                    selectinload(graphs_p.nodes),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                )
                .limit(limit)
                .distinct()
            )

            second_parent = (
                select(graphs_p, graphs_m)
                .join(Dataset, Dataset.dataset_id == graphs_m.dataset_id)
                .join(MixupAttr, MixupAttr.graph_id == graphs_m.graph_id)
                .join(
                    graphs_p, MixupAttr.parent_1_id == graphs_p.graph_id
                )  # join with second parent
                .outerjoin(
                    ged_0,
                    and_(
                        ged_0.graph_0_id == graphs_p.graph_id,
                        ged_0.graph_1_id == graphs_m.graph_id,
                    ),
                )
                .outerjoin(
                    ged_1,
                    and_(
                        ged_1.graph_0_id == graphs_m.graph_id,
                        ged_1.graph_1_id == graphs_p.graph_id,
                    ),
                )
                .where(Dataset.name == dataset_name)
                .where(MixupAttr.mixup_method == method_name)
                .where(
                    and_(
                        ged_0.value.is_(None),
                        ged_1.value.is_(None),
                    )
                )
                .options(
                    selectinload(graphs_m.nodes),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                    selectinload(graphs_p.nodes),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                )
                .limit(limit)
                .distinct()
            )

            both_parents = (
                select(graphs_p, graphs_m)
                .join(Dataset, Dataset.dataset_id == graphs_m.dataset_id)
                .join(MixupAttr, MixupAttr.parent_0_id == graphs_m.graph_id)
                .join(graphs_p, MixupAttr.parent_1_id == graphs_p.graph_id)
                .outerjoin(
                    ged_0,
                    and_(
                        ged_0.graph_0_id == graphs_p.graph_id,
                        ged_0.graph_1_id == graphs_m.graph_id,
                    ),
                )
                .outerjoin(
                    ged_1,
                    and_(
                        ged_1.graph_0_id == graphs_m.graph_id,
                        ged_1.graph_1_id == graphs_p.graph_id,
                    ),
                )
                .where(Dataset.name == dataset_name)
                .where(MixupAttr.mixup_method == method_name)
                .where(graphs_p.graph_id != graphs_m.graph_id)
                .where(
                    and_(
                        ged_0.value.is_(None),
                        ged_1.value.is_(None),
                    )
                )
                .options(
                    selectinload(graphs_p.nodes),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_p.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                    selectinload(graphs_m.nodes),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_m.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                )
                .limit(limit)
                .distinct()
            )

            parents_1 = both_parents.where(
                graphs_p.graph_id <= graphs_m.graph_id
            )
            parents_2 = both_parents.where(
                graphs_m.graph_id < graphs_p.graph_id
            )

            queries = [first_parent, second_parent, parents_1, parents_2]
            empty_counter = 0

            while queries:
                for i, query in enumerate(queries):
                    with Session(self._engine) as session:
                        logger.info(f"Query {i}")
                        res = session.execute(query).all()
                        logger.info(
                            f"Found {len(res)} graph pairs without GED."
                        )

                        if not res:
                            empty_counter += 1
                            logger.info(f"Empty Counter: {empty_counter}")
                            if empty_counter >= len(queries):
                                return
                        else:
                            empty_counter = 0
                            yield [(r.graphs_p, r.graphs_m) for r in res]

    def get_graph_pairs_without_ged(
        self, dataset_name: DatasetName, limit: int
    ) -> Generator[list[tuple[Graph, Graph]], None, None]:
        while True:
            graphs_0 = aliased(Graph, name="graphs_0")
            graphs_1 = aliased(Graph, name="graphs_1")

            stmt = (
                select(graphs_0, graphs_1)
                .join(
                    graphs_1,
                    and_(
                        graphs_0.dataset_id == graphs_1.dataset_id,
                        graphs_0.graph_id < graphs_1.graph_id,
                    ),
                )
                # ===
                # Make sure that no GED exists for neither (graph0, graph1) nor
                #  (graph1, graph0).
                # ===
                .where(
                    ~(
                        select(1)
                        .where(
                            and_(
                                GED.graph_0_id == graphs_0.graph_id,
                                GED.graph_1_id == graphs_1.graph_id,
                            )
                        )
                        .exists()
                    )
                )
                .where(
                    ~(
                        select(1)
                        .where(
                            and_(
                                GED.graph_0_id == graphs_1.graph_id,
                                GED.graph_1_id == graphs_0.graph_id,
                            )
                        )
                        .exists()
                    )
                )
                # ===
                # Exclude mixup graphs.
                # ===
                .where(
                    ~(
                        select(1)
                        .where(MixupAttr.graph_id == graphs_0.graph_id)
                        .exists()
                    )
                )
                .where(
                    ~(
                        select(1)
                        .where(MixupAttr.graph_id == graphs_1.graph_id)
                        .exists()
                    )
                )
                # ===
                # Filter based on dataset name.
                # ===
                .join(Dataset, Dataset.dataset_id == graphs_0.dataset_id)
                .where(Dataset.name == dataset_name)
                # ===
                # Eager load nodes and edges.
                # ===
                .options(
                    selectinload(graphs_0.nodes),
                    selectinload(graphs_0.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_0.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                    selectinload(graphs_1.nodes),
                    selectinload(graphs_1.edges).joinedload(
                        Edge.node_0, innerjoin=True
                    ),
                    selectinload(graphs_1.edges).joinedload(
                        Edge.node_1, innerjoin=True
                    ),
                )
                .limit(limit)
            )

            with Session(self._engine) as session:
                res = session.execute(stmt).all()

            if not res:
                break

            logger.info(f"Found {len(res)} graph pairs without GED.")
            logger.info(f"First pair: {res[0][0].graph_id, res[0][1].graph_id}")

            yield [(r.graphs_0, r.graphs_1) for r in res]
