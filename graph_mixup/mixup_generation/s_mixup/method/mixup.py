import logging

import numpy as np
import pygmtools
import torch
from torch import Tensor
from torch.nn.functional import softmax
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from graph_mixup.mixup_generation.s_mixup.method.gmnet import (
    LitGraphMatchingNet,
)
from graph_mixup.mixup_generation.s_mixup.typing import (
    NormalizeMethod,
    SimMethod,
    SMixupMethodConfig,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def mixup(
    g0: Data,
    g1: Data,
    graph_matching_network: LitGraphMatchingNet,
    config: SMixupMethodConfig,
    *,
    lam: float | None = None,
    sim_method: SimMethod = "cos",
    temperature: float = 1.0,
    normalize_method: NormalizeMethod = "softmax",
    device: int,
) -> Data:
    # Sample mixup lambda
    if lam is None:
        lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)

    lam = max(lam, 1 - lam)

    # Mimic batching
    batch0 = Batch.from_data_list([g0]).to(device)
    batch1 = Batch.from_data_list([g1]).to(device)

    # Compute node embeddings from the graph matching network
    with torch.no_grad():
        graph_matching_network.eval()
        graph_matching_network.to(device)
        h0, h1 = graph_matching_network.model.encoder(
            batch0, batch1, readout=False
        )
        h0, h1 = h0.to("cpu"), h1.to("cpu")

    # Compute matching matrix
    if sim_method == "cos":
        emb1 = h0 / h0.norm(dim=1)[:, None]
        emb2 = h1 / h1.norm(dim=1)[:, None]
        match = emb1 @ emb2.T / temperature
    elif sim_method == "abs_diff":
        match = -(h0.unsqueeze(1) - h1.unsqueeze(0)).norm(dim=-1)

    # Normalize matching matrix
    if normalize_method == "softmax":
        normalized_match = softmax(match.clone(), dim=0)
    elif normalize_method == "sinkhorn":
        normalized_match = pygmtools.sinkhorn(
            match.clone(),
            backend="pytorch",
        )

    # Compute mixup adjacency matrix
    mixed_adj = (
        lam * to_dense_adj(g0.edge_index)[0].double()
        + (1 - lam)
        * normalized_match.double()
        @ to_dense_adj(g1.edge_index)[0].double()
        @ normalized_match.double().T
    )
    mixed_adj[mixed_adj < 0.1] = 0

    # Compute mixup node features
    mixed_x = lam * g0.x + (1 - lam) * normalized_match.float() @ g1.x

    # Compute mixup label
    mixed_y = lam * g0.y + (1 - lam) * g1.y

    # Create and return PyG data object
    edge_index, edge_weight = dense_to_sparse(mixed_adj)
    return Data(
        x=mixed_x.float(),
        y=mixed_y.float(),
        edge_index=edge_index,
        edge_weight=edge_weight.to(torch.float),
    )


@torch.no_grad()
def batched_mixup(
    batch0: Batch,
    batch1: Batch,
    graph_matching_network: LitGraphMatchingNet,
    config: SMixupMethodConfig,
    *,
    lambdas: np.ndarray | None = None,
    sim_method: SimMethod = "cos",
    temperature: float = 1.0,
    normalize_method: NormalizeMethod = "softmax",
    device: int,
) -> list[Data]:
    assert batch0.num_graphs == batch1.num_graphs

    # Sample mixup lambda.
    if lambdas is None:
        lambdas = np.random.beta(
            config.mixup_alpha, config.mixup_alpha, batch0.num_graphs
        )

    lambdas = np.max([lambdas, 1 - lambdas], axis=0)

    # noinspection PyUnresolvedReferences
    batch0, batch1 = batch0.to(device), batch1.to(device)

    # Compute node embeddings from the graph matching network
    graph_matching_network.eval()
    graph_matching_network.to(device)
    z0, z1 = graph_matching_network.model.encoder(batch0, batch1, readout=False)
    z0, z1 = z0.to("cpu"), z1.to("cpu")
    batch0, batch1 = batch0.to("cpu"), batch1.to("cpu")

    embedding_pairs: list[tuple[Tensor, Tensor]] = []
    for i in range(batch0.num_graphs):
        emb0 = z0[
            batch0._slice_dict["x"][i] : batch0._slice_dict["x"][i + 1], :
        ]
        emb1 = z1[
            batch1._slice_dict["x"][i] : batch1._slice_dict["x"][i + 1], :
        ]
        embedding_pairs.append((emb0, emb1))

    input_graph_pairs = zip(batch0.to_data_list(), batch1.to_data_list())
    mixed_graphs: list[Data] = []

    for i, (graphs, embeddings) in enumerate(
        zip(input_graph_pairs, embedding_pairs)
    ):
        try:
            g0, g1 = graphs
            h0, h1 = embeddings
            lam = lambdas[i]
            # Compute matching matrix
            if sim_method == "cos":
                h0_normalized = h0 / h0.norm(dim=1)[:, None]
                h1_normalized = h1 / h1.norm(dim=1)[:, None]
                match = h0_normalized @ h1_normalized.T / temperature
            elif sim_method == "abs_diff":
                match = -(h0.unsqueeze(1) - h1.unsqueeze(0)).norm(dim=-1)

            # Normalize matching matrix
            if normalize_method == "softmax":
                normalized_match = softmax(match.clone(), dim=0)
            elif normalize_method == "sinkhorn":
                normalized_match = pygmtools.sinkhorn(
                    match.clone(),
                    backend="pytorch",
                )

            # Compute mixup adjacency matrix
            mixed_adj = (
                lam * to_dense_adj(g0.edge_index)[0].double()
                + (1 - lam)
                * normalized_match.double()
                @ to_dense_adj(g1.edge_index)[0].double()
                @ normalized_match.double().T
            )
            mixed_adj[mixed_adj < 0.1] = 0

            # Compute mixup node features
            mixed_x = lam * g0.x + (1 - lam) * normalized_match.float() @ g1.x

            # Compute mixup label
            mixed_y = lam * g0.y + (1 - lam) * g1.y

            # Create and return PyG data object
            edge_index, edge_weight = dense_to_sparse(mixed_adj)

            mixed_graphs.append(
                Data(
                    x=mixed_x.float(),
                    y=mixed_y.float(),
                    edge_index=edge_index,
                    edge_weight=edge_weight.to(torch.float),
                )
            )
        except RuntimeError as e:
            logger.warning(f"S-Mixup Error: {e}")

    return mixed_graphs
