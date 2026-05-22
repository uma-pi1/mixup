from abc import abstractmethod

from graph_mixup.compute_ged.typing import GEDResult
from graph_mixup.compute_ged.computers.utils import _lower_bound
from graph_mixup.ged_database.models import Graph


class AbstractGEDComputer:
    def __init__(self, timeout: int, lb_threshold: int) -> None:
        self.timeout = timeout
        self.lb_threshold = lb_threshold

    def process(self, g0: Graph, g1: Graph) -> GEDResult:
        # ===
        # Compute a lower bound. If it is above the threshold, skip GED
        # computation.
        # ===

        computed_lb = _lower_bound(g0, g1)
        if computed_lb > self.lb_threshold:
            return GEDResult(g0.graph_id, g1.graph_id, -1, 0, None, computed_lb)

        # ===
        # Check that g0.num_nodes <= g1.num_nodes. If not, swap g0 and g1.
        # Reason: GED binary will swap graphs otherwise by itself without
        #  notification. If this occurred, the mapping would be inverse.
        # ===

        if g0.num_nodes() <= g1.num_nodes():
            return self._compute(g0, g1, computed_lb)
        else:
            return self._compute(g1, g0, computed_lb)

    @abstractmethod
    def _compute(self, g0: Graph, g1: Graph, lb: int) -> GEDResult: ...
