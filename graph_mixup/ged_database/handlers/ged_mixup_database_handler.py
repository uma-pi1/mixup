import logging
from typing import Any, Generator

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased, selectinload

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.models import (
    GED,
    Dataset,
    Edge,
    Graph,
    MixupAttr,
    Node,
)
from graph_mixup.ged_mixup.typing import TempEdge, TempNode

logger = logging.getLogger(__name__)


class GEDMixupDatabaseHandler(BaseHandler):
    def get_graph_pairs_with_ged(
        self,
        dataset_name: DatasetName,
        *,
        batch_size: int = 32,
        exclude_mixup: bool = False,
    ) -> Generator[list[tuple[Graph, Graph, GED]], None, None]:
        offset = 0

        graphs_0 = aliased(Graph, name="graphs_0")
        graphs_1 = aliased(Graph, name="graphs_1")
        geds = aliased(GED, name="geds")
        datasets_0 = aliased(Dataset, name="datasets_0")
        datasets_1 = aliased(Dataset, name="datasets_1")
        mixup_attrs_0 = aliased(MixupAttr, name="mixup_attrs_0")
        mixup_attrs_1 = aliased(MixupAttr, name="mixup_attrs_1")

        while True:
            stmt = (
                select(graphs_0, graphs_1, geds)
                .join(graphs_0, geds.graph_0_id == graphs_0.graph_id)
                .join(graphs_1, geds.graph_1_id == graphs_1.graph_id)
                .join(datasets_0, datasets_0.dataset_id == graphs_0.dataset_id)
                .join(datasets_1, datasets_1.dataset_id == graphs_1.dataset_id)
                .where(
                    datasets_0.name == dataset_name,
                    datasets_1.name == dataset_name,
                )
                .where(geds.value > -1)
                .options(
                    selectinload(geds.mappings),
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
                .order_by(func.rand())
                .limit(batch_size)
                .offset(offset)
            )

            if exclude_mixup:
                stmt = (
                    stmt.outerjoin(
                        mixup_attrs_0,
                        mixup_attrs_0.graph_id == graphs_0.graph_id,
                    )
                    .outerjoin(
                        mixup_attrs_1,
                        mixup_attrs_1.graph_id == graphs_1.graph_id,
                    )
                    .where(
                        mixup_attrs_0.mixup_attr_id.is_(None),
                        mixup_attrs_1.mixup_attr_id.is_(None),
                    )
                )

            with Session(self._engine) as session:
                results = session.execute(stmt).all()

            if not results:
                break

            yield [(row.graphs_0, row.graphs_1, row.geds) for row in results]
            offset += batch_size

    def create_ged_mixup_graph(
        self,
        dataset_id: int,
        graph_0_id: int,
        graph_1_id: int,
        lam: float | None,
        mixup_path_len: int,
        ged: int,
        nodes: dict[int, TempNode],
        edges: dict[tuple[int, int], TempEdge],
        label: list[float],
        mixup_hyperparameters: dict[str, Any],
        creation_time_us: int,
    ) -> Graph | None:
        with Session(self._engine) as session, session.begin():
            # ===
            # Add graph
            # ===
            graph_sql = Graph(
                dataset_id=dataset_id,
                label=label,
            )
            session.add(graph_sql)
            session.flush()
            graph_id = graph_sql.graph_id

            # ===
            # Add nodes
            # ===
            nodes_sql: dict[int, Node] = dict()
            for i, (prev_id, node) in enumerate(nodes.items()):
                if node.attributes is not None:
                    nodes_sql[prev_id] = Node(
                        graph_id=graph_sql.graph_id,
                        index=i,
                        attributes=node.attributes,
                    )
                else:
                    # ===
                    # Prevent JSON null (i.e., use SQL null instead).
                    # ===
                    nodes_sql[prev_id] = Node(
                        graph_id=graph_sql.graph_id, index=i
                    )
            session.add_all(nodes_sql.values())
            session.flush()

            # ===
            # Add edges
            # ===
            edges_sql: list[Edge] = []
            for i, edge in enumerate(edges.values()):
                # Obtain correct order of nodes.
                id0 = nodes_sql[edge.id0].node_id
                id1 = nodes_sql[edge.id1].node_id
                if id0 < id1:
                    smaller_id, larger_id = id0, id1
                else:
                    smaller_id, larger_id = id1, id0

                # Create DB entity.
                edges_sql.append(
                    Edge(
                        graph_id=graph_sql.graph_id,
                        index=i,
                        node_0_id=smaller_id,
                        node_1_id=larger_id,
                        attributes=edge.attributes,
                    )
                )
            session.add_all(edges_sql)

            # ===
            # Add MixupAttr
            # ===
            session.add(
                MixupAttr(
                    graph_id=graph_id,
                    parent_0_id=graph_0_id,
                    parent_1_id=graph_1_id,
                    mixup_lambda=lam,
                    mixup_hyperparameters=mixup_hyperparameters,
                    mixup_method="ged_mixup",
                    creation_time_us=creation_time_us,
                )
            )
            # ===
            # Add GEDs (twice: parent <-> mixup child)
            # ===
            session.add(
                GED(
                    graph_0_id=graph_0_id,
                    graph_1_id=graph_id,
                    value=mixup_path_len,
                )
            )

            session.add(
                GED(
                    graph_0_id=graph_1_id,
                    graph_1_id=graph_id,
                    value=ged - mixup_path_len,
                )
            )

        with Session(self._engine) as session:
            stmt = select(Graph).where(Graph.graph_id == graph_id)
            return session.execute(stmt).scalar_one()
