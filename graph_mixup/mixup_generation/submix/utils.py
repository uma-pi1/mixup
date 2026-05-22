import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# ===
# Public.
# ===


class NoEdgesException(Exception): ...


def get_cc_sizes(graph):
    nn_sets = [set(e) for e in _get_neighbors(graph)]
    num_nodes = graph.num_nodes
    out = np.zeros(num_nodes, dtype=int)
    seen = set()
    for u in range(num_nodes):
        if u not in seen:
            cc = _run_bfs(nn_sets, u)
            seen.update(cc)
            for v in cc:
                out[v] = len(cc)
    return out


def make_subgraph(graph, root, cc_sizes, max_iters=10):
    diffuse = _PPR()
    out = torch.zeros(graph.num_nodes)
    out[root] = 1.0
    for _ in range(max_iters):
        out = diffuse(out, graph.edge_index)
    return out.argsort(descending=True)[: cc_sizes[root]]


def mix_graphs(
    dst_graph, dst_nodes, src_graph, src_nodes, label_by="edges"
) -> tuple[Data, float]:
    node_map = torch.zeros(src_graph.num_nodes, dtype=torch.long)
    node_map[src_nodes] = dst_nodes
    dst_mask = _to_mask(dst_graph.num_nodes, dst_nodes)
    src_mask = _to_mask(src_graph.num_nodes, src_nodes)

    edges1 = dst_graph.edge_index
    edges1 = edges1[:, ~(dst_mask[edges1[0]] & dst_mask[edges1[1]])]
    edges2 = src_graph.edge_index
    edges2 = node_map[edges2[:, src_mask[edges2[0]] & src_mask[edges2[1]]]]

    new_x = dst_graph.x.clone()
    new_x[dst_nodes] = src_graph.x[src_nodes]
    new_y = torch.zeros(dst_graph.y.size(1))
    new_y[_one_hot_to_idx(dst_graph.y)] = 1

    new_edges = torch.cat([edges1, edges2], dim=1)
    if new_edges.size(1) == 0:
        raise NoEdgesException("mixup graph contains no edges")

    if label_by == "nodes":
        ratio = len(dst_nodes) / dst_graph.num_nodes
    elif label_by == "edges":
        ratio = edges1.size(1) / new_edges.size(1)
    else:
        raise ValueError(label_by)

    new_y[_one_hot_to_idx(dst_graph.y)] *= ratio
    new_y[_one_hot_to_idx(src_graph.y)] += 1 - ratio
    return Data(new_x, new_edges, y=new_y.unsqueeze(0)), ratio


# ===
# Private.
# ===


class _PPR(MessagePassing):
    def __init__(self, self_loops=True, alpha=0.15):
        super().__init__()
        self.alpha = alpha
        self.self_loops = self_loops
        self.edge_index = None

    def forward(self, x, edge_index):
        if self.edge_index is None:
            self.edge_index = gcn_norm(
                edge_index, add_self_loops=self.self_loops
            )
        edge_index, edge_weight = self.edge_index
        out = self.propagate(
            edge_index, x=x.view(-1, 1), edge_weight=edge_weight
        )
        out = out.view(-1) / out.sum()
        return self.alpha * x + (1 - self.alpha) * out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


def _run_bfs(neighbor_sets, node):
    seen = set()
    next_level = {node}
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in seen:
                seen.add(v)
                next_level.update(neighbor_sets[v])
    return seen


def _get_neighbors(graph):
    # Assume the edge_index is undirected and sorted.
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.numpy()

    diff = np.diff(np.insert(edge_index[0, :], 0, 0))
    start_index = np.full(num_nodes, fill_value=-1, dtype=np.int64)
    start_index[(diff[diff > 0]).cumsum()] = np.nonzero(diff)[0]
    start_index[0] = 0
    for i in range(num_nodes - 1, 0, -1):
        if start_index[i] < 0:
            if i == num_nodes - 1:
                start_index[i] = num_nodes
            else:
                start_index[i] = start_index[i + 1]

    neighbors = []
    for node in range(num_nodes):
        if node < num_nodes - 1:
            end_index = start_index[node + 1]
        else:
            end_index = edge_index.shape[1]
        neighbors.append(edge_index[1, start_index[node] : end_index])
    return neighbors


def _to_mask(num_nodes, subset):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[subset] = 1
    return mask


def _one_hot_to_idx(one_hot: Tensor) -> int:
    return one_hot.argmax(dim=1).item()
