from statistics import median

import torch
from torch import Tensor
from torch.distributions.beta import Beta
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import degree, dense_to_sparse

from graph_mixup.mixup_generation.g_mixup.typing import GMixupConfig


def compute_median_node_number(dataset: list[Data]):
    num_nodes = []
    for data in dataset:
        num_nodes.append(data.num_nodes)
    return median(num_nodes)


def split_dataset_into_classes(dataset: list[Data]) -> dict[int, list[Data]]:
    # Assume: Graph labels are one-hot encoded!
    assert dataset[0].y.dim() > 1, "graph labels must be one-hot encoded"

    splits: dict[int, list[Data]] = {
        cls: [] for cls in range(dataset[0].y.size(1))
    }
    for data in dataset:
        cls = int(torch.argmax(data.y, dim=-1).item())
        splits[cls].append(data)
    return splits


def align_graphs(
    graphs: list[Data], target_size: int
) -> tuple[list[Tensor], list[Tensor]]:
    adj_matrices, feat_matrices = [], []
    for data in graphs:
        # Sort node indices descendingly based upon degree
        node_degs = degree(data.edge_index[0])
        sorted_indices = torch.argsort(node_degs, descending=True)

        # Decrease to target shape (if currently larger)
        sorted_indices = sorted_indices[:target_size]

        # FEATURE MATRIX

        assert data.x is not None, "graphs need features"
        sorted_features = data.x[sorted_indices]

        # Increase to target length by zero-padding (if smaller)
        missing = target_size - len(sorted_features)
        if missing > 0:
            sorted_features = torch.cat(
                [sorted_features, torch.zeros(missing, data.num_features)]
            )

        feat_matrices.append(sorted_features)

        # ADJACENCY MATRIX

        # Create sparse adj matrix
        edge_index_sp = torch.sparse_coo_tensor(
            indices=data.edge_index,
            values=torch.ones(
                data.edge_index.size(1)
            ),  # edge_index: [2, num_edges]
            size=(len(node_degs), len(node_degs)),
        )

        # Permute rows and columns of adj matrix
        sorted_edge_index_sp = torch.index_select(
            torch.index_select(edge_index_sp, dim=0, index=sorted_indices),
            dim=1,
            index=sorted_indices,
        )

        # Increase to target size (if currently smaller)
        sorted_edge_index_sp.sparse_resize_(
            (target_size, target_size), 2, 0
        )  # in-place op

        adj_matrices.append(sorted_edge_index_sp)

    return adj_matrices, feat_matrices


def universal_singular_value_thresholding(
    aligned_graphs: list[Tensor], threshold=2.02
) -> Tensor:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values (should be larger than 2 to be mathematically sound)

    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs_stack = torch.stack(aligned_graphs)
    num_graphs = aligned_graphs_stack.size(0)

    sum_graph = torch.sparse.sum(aligned_graphs_stack, dim=0)
    mean_graph = (sum_graph / num_graphs).to_dense()

    num_nodes = mean_graph.size(0)

    u, s, v = torch.svd(mean_graph)
    singular_threshold = threshold * (num_nodes**0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    return graphon


def compute_average_feat_matrix(feat_matrices: list[Tensor]) -> Tensor:
    stack = torch.stack(feat_matrices)
    return stack.mean(dim=0)


def compute_class_graphons_and_features(
    dataset: list[Data], threshold=0.2
) -> dict[int, dict[str, Tensor]]:
    # Assume: Graph labels are one-hot encoded!
    assert dataset[0].y.dim() > 1, "graph labels must be one-hot encoded"

    resolution = int(compute_median_node_number(dataset))
    class_graphs = split_dataset_into_classes(dataset)

    # Compute graphon for each class
    classes = dict()
    for label, graphs in class_graphs.items():
        adj_matrices, feat_matrices = align_graphs(graphs, resolution)

        graphon = universal_singular_value_thresholding(adj_matrices, threshold)
        feat_matrix = compute_average_feat_matrix(feat_matrices)
        one_hot_label = one_hot(torch.tensor([label]), len(class_graphs))

        classes[label] = dict(
            graphon=graphon,
            feat_matrix=feat_matrix,
            one_hot_label=one_hot_label,
        )

    return classes


def sample_adj_matrix(graphon: Tensor) -> Tensor:
    # Sample from mixup graphon
    sampled_matrix = torch.rand(*graphon.shape).le(graphon).int()
    adj_matrix = sampled_matrix.triu() + sampled_matrix.triu().t()

    # Zero-out diagonal
    num_nodes = graphon.size(0)
    mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool) ^ torch.eye(
        num_nodes, dtype=torch.bool
    )
    return torch.where(mask, adj_matrix, torch.zeros(num_nodes, num_nodes))


def sample_graph(class_pair: tuple[dict, dict], config: GMixupConfig):
    assert not class_pair[0]["one_hot_label"].equal(
        class_pair[1]["one_hot_label"]
    ), "classes must be different"

    # Sample mixup parameter
    distr = Beta(config.mixup_alpha, config.mixup_alpha)
    lam = distr.sample()

    # Compute mixup
    graphon = (
        lam * class_pair[0]["graphon"] + (1 - lam) * class_pair[1]["graphon"]
    )
    feat_matrix: Tensor = (
        lam * class_pair[0]["feat_matrix"]
        + (1 - lam) * class_pair[1]["feat_matrix"]
    )
    label = (
        lam * class_pair[0]["one_hot_label"]
        + (1 - lam) * class_pair[1]["one_hot_label"]
    )

    # Sample adjacency matrix
    adj_matrix = sample_adj_matrix(graphon)
    edge_index, _ = dense_to_sparse(adj_matrix)

    # create new graph
    return Data(
        x=feat_matrix,
        edge_index=edge_index,
        y=label,
    )
