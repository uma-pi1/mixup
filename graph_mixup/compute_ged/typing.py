from dataclasses import dataclass
from enum import Enum


class ComputationMode(str, Enum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"


class GEDLIBMethod(str, Enum):
    BRANCH = "BRANCH"
    BRANCH_FAST = "BRANCH_FAST"
    BRANCH_TIGHT = "BRANCH_TIGHT"
    BRANCH_UNIFORM = "BRANCH_UNIFORM"
    BRANCH_COMPACT = "BRANCH_COMPACT"
    PARTITION = "PARTITION"
    HYBRID = "HYBRID"
    RING = "RING"
    ANCHOR_AWARE_GED = "ANCHOR_AWARE_GED"
    WALKS = "WALKS"
    IPFP = "IPFP"
    BIPARTITE = "BIPARTITE"
    SUBGRAPH = "SUBGRAPH"
    NODE = "NODE"
    REFINE = "REFINE"
    BP_BEAM = "BP_BEAM"
    SIMULATED_ANNEALING = "SIMULATED_ANNEALING"
    HED = "HED"
    STAR = "STAR"


@dataclass
class GEDResult:
    graph_0_id: int
    graph_1_id: int
    value: int
    time: int | None
    mapping: dict[int, int] | None
    lb: int


class MissingGEDException(Exception): ...


class MissingMappingException(Exception): ...
