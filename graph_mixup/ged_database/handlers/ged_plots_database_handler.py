from dataclasses import dataclass
from typing import Generator

from sqlalchemy import Select, and_, func, select
from sqlalchemy.orm import Session, aliased

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.models import (
    GED,
    Dataset,
    Graph,
    MixupAttr,
)


@dataclass
class MixupGEDTriple:
    parents_ged: int
    parent_child_geds: tuple[int, int]


class GEDPlotsDatabaseHandler(BaseHandler):
    def get_mixup_ged_triples(
        self,
        dataset_name: DatasetName,
        method_name: str,
        lam: float,
        limit: int,
        tol: float,
    ) -> Generator[list[MixupGEDTriple], None, None]:
        # ===
        # Define table aliases.
        # ===
        geds1 = aliased(GED, name="geds1")
        geds1_rev = aliased(GED, name="geds1_rev")
        geds2 = aliased(GED, name="geds2")
        geds2_rev = aliased(GED, name="geds2_rev")
        geds3 = aliased(GED, name="geds3")
        geds3_rev = aliased(GED, name="geds3_rev")

        def stmt(offset: int) -> Select:
            return (
                select(
                    MixupAttr.graph_id,
                    MixupAttr.parent_0_id,
                    MixupAttr.parent_1_id,
                    func.coalesce(geds1.value, geds1_rev.value).label(
                        "mixup_parent_0_ged"
                    ),
                    func.coalesce(geds2.value, geds2_rev.value).label(
                        "mixup_parent_1_ged"
                    ),
                    func.coalesce(geds3.value, geds3_rev.value).label(
                        "parents_ged"
                    ),
                )
                .outerjoin(
                    geds1,
                    and_(
                        MixupAttr.graph_id == geds1.graph_0_id,
                        MixupAttr.parent_0_id == geds1.graph_1_id,
                    ),
                )
                .outerjoin(
                    geds1_rev,
                    and_(
                        MixupAttr.graph_id == geds1_rev.graph_1_id,
                        MixupAttr.parent_0_id == geds1_rev.graph_0_id,
                    ),
                )
                .outerjoin(
                    geds2,
                    and_(
                        MixupAttr.graph_id == geds2.graph_0_id,
                        MixupAttr.parent_1_id == geds2.graph_1_id,
                    ),
                )
                .outerjoin(
                    geds2_rev,
                    and_(
                        MixupAttr.graph_id == geds2_rev.graph_1_id,
                        MixupAttr.parent_1_id == geds2_rev.graph_0_id,
                    ),
                )
                .outerjoin(
                    geds3,
                    and_(
                        MixupAttr.parent_0_id == geds3.graph_0_id,
                        MixupAttr.parent_1_id == geds3.graph_1_id,
                    ),
                )
                .outerjoin(
                    geds3_rev,
                    and_(
                        MixupAttr.parent_0_id == geds3_rev.graph_1_id,
                        MixupAttr.parent_1_id == geds3_rev.graph_0_id,
                    ),
                )
                .join(Graph, Graph.graph_id == MixupAttr.graph_id)
                .join(Dataset, Dataset.dataset_id == Graph.dataset_id)
                .where(Dataset.name == dataset_name)
                .where(MixupAttr.mixup_method == method_name)
                .where(func.abs(MixupAttr.mixup_lambda - lam) <= tol)
                .where(
                    func.coalesce(geds1.value, geds1_rev.value).isnot(None),
                    func.coalesce(geds2.value, geds2_rev.value).isnot(None),
                    func.coalesce(geds3.value, geds3_rev.value).isnot(None),
                    func.coalesce(geds1.value, geds1_rev.value) > -1,
                    func.coalesce(geds2.value, geds2_rev.value) > -1,
                    func.coalesce(geds3.value, geds3_rev.value) > -1,
                )
                .limit(limit)
                .offset(offset)
            )

        offset = 0

        while True:
            with Session(self._engine) as session:
                result_rows = session.execute(stmt(offset)).all()

            if not result_rows:
                break

            yield [
                MixupGEDTriple(
                    parents_ged=row.parents_ged,
                    parent_child_geds=(
                        row.mixup_parent_0_ged,
                        row.mixup_parent_1_ged,
                    ),
                )
                for row in result_rows
            ]

            offset += limit
