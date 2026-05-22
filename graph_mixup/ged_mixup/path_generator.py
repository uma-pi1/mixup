from random import shuffle, seed
from typing import Generator

from graph_mixup.ged_database.models import Graph
from graph_mixup.ged_mixup.typing import (
    NodeRelabelOp,
    EdgeDeletionOp,
    NodeInsertionOp,
    EdgeInsertionOp,
    PathOps,
    EditOp,
)


class PathGenerator:
    def __init__(
        self, g0: Graph, g1: Graph, mappings: dict[int, int], seed: int
    ) -> None:
        assert g0.num_nodes() <= g1.num_nodes()

        self.seed = seed
        self.smaller_graph = g0
        self.mappings = mappings

        self._path_ops = self._get_path_ops(g0, g1, mappings)
        self._n = len(self._path_ops)

    def generate(self) -> Generator[list[EditOp], None, None]:
        all_available_ops = list(self._path_ops.get_all())
        seed(self.seed)
        shuffle(all_available_ops)

        def backtrack(
            path: list[EditOp],
            added_ops: set[EditOp],
            inserted_node_ids: set[int],
        ) -> Generator[list[EditOp], None, None]:
            if len(path) == self._n:
                yield path
                return

            for addition in all_available_ops:
                if addition not in added_ops:
                    # Keep track of operations and added node ids.
                    new_added_ops = added_ops.copy()
                    new_added_ops.add(addition)
                    new_inserted_node_ids = inserted_node_ids.copy()
                    if isinstance(addition, NodeInsertionOp):
                        new_inserted_node_ids.add(addition.image_node_id)

                    # ===
                    # If addition is EdgeInsertionOp: Addition might require
                    # insertions of some nodes first (resp. ids stored in
                    # the attribute required_image_node_ids). If those nodes
                    # have not been inserted, addition is inadmissible.
                    #
                    # Otherwise: Addition should cause no trouble.
                    # ===

                    if not isinstance(addition, EdgeInsertionOp):
                        yield from backtrack(
                            path + [addition],
                            new_added_ops,
                            new_inserted_node_ids,
                        )
                    else:
                        if addition.required_image_node_ids.issubset(
                            new_inserted_node_ids
                        ):
                            yield from backtrack(
                                path + [addition],
                                new_added_ops,
                                new_inserted_node_ids,
                            )

        yield from backtrack([], set(), set())

    @staticmethod
    def _get_path_ops(
        g0: Graph, g1: Graph, mappings: dict[int, int]
    ) -> PathOps:
        touched_image_nodes: set[int] = set()

        # ===
        # Apply mapping: Relabel nodes if required.
        # ===

        larger_graph_nodes_dict = {node.node_id: node for node in g1.nodes}
        node_relabel_ops: set[NodeRelabelOp] = set()
        for node in g0.nodes:
            # Apply the mapping to get the larger graph's node
            image_id = mappings[node.node_id]
            touched_image_nodes.add(image_id)
            mapped_node = larger_graph_nodes_dict[image_id]

            # Relabel the node if necessary
            if mapped_node.attributes != node.attributes:
                node_relabel_ops.add(
                    NodeRelabelOp(
                        node.node_id,
                        mapped_node.node_id,
                        tuple(mapped_node.attributes)
                        if mapped_node.attributes is not None
                        else None,
                    )
                )

        # ===
        # Compare existing edges w/ required edges: Delete unwanted existing
        # edges.
        # ===

        larger_graph_edges = {
            (edge.node_0_id, edge.node_1_id): edge for edge in g1.edges
        }
        touched_edges: set[tuple[int, int]] = set()
        edge_deletion_ops: set[EdgeDeletionOp] = set()

        for edge in g0.edges:
            node0 = mappings[edge.node_0_id]
            touched_image_nodes.add(node0)
            node1 = mappings[edge.node_1_id]
            touched_image_nodes.add(node1)

            if (node0, node1) in larger_graph_edges:
                touched_edges.add((node0, node1))
            elif (node1, node0) in larger_graph_edges:
                touched_edges.add((node1, node0))
            else:
                edge_deletion_ops.add(
                    EdgeDeletionOp(edge.node_0_id, edge.node_1_id)
                )

        # ===
        # Compare required edges w/ existing edges: Add missing edges.
        # ===

        untouched_edges = {
            (edge.node_0_id, edge.node_1_id): edge
            for edge in g1.edges
            if (edge.node_0_id, edge.node_1_id) not in touched_edges
        }

        inverse_mapping: dict[int, int] = {v: k for k, v in mappings.items()}
        # maps: image -> preimage, hence "keys == image".

        node_insertion_ops: set[NodeInsertionOp] = set()
        edge_insertion_ops: set[EdgeInsertionOp] = set()
        for (id0, id1), edge in untouched_edges.items():
            touched_image_nodes.add(id0)
            touched_image_nodes.add(id1)

            # ===
            # First: Check whether both endpoints exist.
            # ===
            #
            #  For that, check whether a preimage exists for each endpoint. This
            #  is equivalent to checking whether the endpoint is in the image of
            #  the mapping (since in this case there must be a preimage).
            #
            #  If there is no preimage for an endpoint, then this node must be
            #  inserted.

            required_ids: list[int] = []

            if id0 not in inverse_mapping:
                required_ids.append(id0)
                node_insertion_ops.add(
                    NodeInsertionOp(
                        id0,
                        tuple(edge.node_0.attributes)
                        if edge.node_0.attributes is not None
                        else None,
                    )
                )
            else:
                # Node is in image which means that there is a preimage. Use
                # preimage id as id0.
                id0 = inverse_mapping[id0]

            if id1 not in inverse_mapping:
                required_ids.append(id1)
                node_insertion_ops.add(
                    NodeInsertionOp(
                        id1,
                        tuple(edge.node_1.attributes)
                        if edge.node_1.attributes is not None
                        else None,
                    )
                )
            else:
                # (see above)
                id1 = inverse_mapping[id1]

            # ===
            # Then: Insert edge.
            # ===

            edge_insertion_ops.add(
                EdgeInsertionOp(
                    id0,
                    id1,
                    (
                        tuple(edge.attributes)
                        if edge.attributes is not None
                        else None
                    ),
                    frozenset(required_ids),
                )
            )

        # ===
        # Compare touched nodes w/ all required nodes: Add untouched nodes.
        # ===

        untouched_nodes = {
            node.node_id for node in g1.nodes
        } - touched_image_nodes

        for node_id in untouched_nodes:
            node = larger_graph_nodes_dict[node_id]
            node_insertion_ops.add(
                NodeInsertionOp(
                    node_id,
                    tuple(node.attributes)
                    if node.attributes is not None
                    else None,
                )
            )

        # ===
        # Verify GED: Should be equal to the total number of ops.
        # ===

        path_ops = PathOps(
            node_relabel_ops=node_relabel_ops,
            edge_deletion_ops=edge_deletion_ops,
            node_insertion_ops=node_insertion_ops,
            edge_insertion_ops=edge_insertion_ops,
        )

        return path_ops
