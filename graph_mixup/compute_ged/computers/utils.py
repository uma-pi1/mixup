import os
from collections import Counter
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree

from torch_geometric.data import Data

from graph_mixup.ged_database.models import Graph

# ===
# Utilities for lower bound computation.
# ===


def _num_node_attrs_diff(
    g0_attrs: dict[int, tuple[float, ...]],
    g1_attrs: dict[int, tuple[float, ...]],
) -> int:
    if len(g0_attrs) <= len(g1_attrs):
        smaller_attrs = g0_attrs
        larger_attrs = g1_attrs
    else:
        smaller_attrs = g1_attrs
        larger_attrs = g0_attrs

    lb_relabel_ops = 0

    # Count each attribute in the larger graph.
    larger_counter = Counter(larger_attrs.values())

    # ===
    # A lower bound of the number of node relabel ops is given as the
    # difference in node labels between the smaller graph and the larger
    # graph.
    # ===

    for attr in smaller_attrs.values():
        if larger_counter[attr] == 0:  # Counter is 0 for missing keys.
            lb_relabel_ops += 1
        else:
            larger_counter[attr] -= 1

    return lb_relabel_ops


def _lower_bound(g0: Graph, g1: Graph) -> int:
    num_nodes_diff = abs(g0.num_nodes() - g1.num_nodes())
    num_edges_diff = abs(g0.num_edges() - g1.num_edges())
    num_node_attrs_diff = _num_node_attrs_diff(
        g0.node_attributes_with_default_value(),
        g1.node_attributes_with_default_value(),
    )

    return num_nodes_diff + num_edges_diff + num_node_attrs_diff


# ===
# Utilities for exact GED computation.
# ===


def _make_temp_file(g: Graph) -> str:
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(g.get_ged_library_format())
        return f.name


def _remove_temp_file(name: str) -> None:
    os.remove(name)


# ===
# Utilities for approximate GED computation.
# ===


def _write_gxl(data: Data, file_path: str, graph_id: str = "graph") -> None:
    gxl_root = ElementTree.Element("gxl")
    graph_elem = ElementTree.SubElement(
        gxl_root, "graph", id=graph_id, edgeids="false", edgemode="undirected"
    )

    # Determine the number of nodes.
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = data.num_nodes
    elif data.x is not None:
        num_nodes = data.x.size(0)
    else:
        num_nodes = 0

    for i in range(num_nodes):
        ElementTree.SubElement(graph_elem, "node", id=f"_{i}")

    # Add edges based on data.edge_index.
    try:
        edge_index = data.edge_index
        edge_index = edge_index.tolist()
    except Exception as e:
        print("Error extracting edge_index:", e)
        edge_index = []

    if edge_index and len(edge_index) == 2:
        sources, targets = edge_index
        for src, tgt in zip(sources, targets):
            u, v = int(src), int(tgt)
            # Only write each undirected edge once (PyG stores both directions)
            if u <= v:
                ElementTree.SubElement(
                    graph_elem, "edge", attrib={"from": f"_{u}", "to": f"_{v}"}
                )

    xml_str = ElementTree.tostring(gxl_root, encoding="unicode")

    header = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd">\n'
    )

    with open(file_path, "w") as f:
        f.write(header + xml_str)


def _write_xml_collection(gxl_filenames: list[str], file_path: str) -> None:
    root = ElementTree.Element("GraphCollection")
    for filename in gxl_filenames:
        ElementTree.SubElement(root, "graph", file=filename, **{"class": "a"})

    xml_str = ElementTree.tostring(root, encoding="unicode")
    header = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE GraphCollection SYSTEM "http://www.inf.unibz.it/~blumenthal/dtd/GraphCollection.dtd">\n'
    )
    with open(file_path, "w") as f:
        f.write(header + xml_str)
