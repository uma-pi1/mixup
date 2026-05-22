import logging
import random
import time
from typing import Generic, assert_never

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module import (
    AbstractDataModule,
)
from graph_mixup.augmentations.data.typing import (
    DataModuleConfigType,
    SMixupItem,
)
from graph_mixup.augmentations.typing import AugDatasetConfig
from graph_mixup.config.typing import PreBatchMixupName
from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.handlers.mixup_graph_fetcher import (
    MixupGraphFetcher,
)
from graph_mixup.mixup_generation.g_mixup.graphons import (
    compute_class_graphons_and_features,
    sample_graph,
)
from graph_mixup.mixup_generation.s_mixup.method.gmnet import (
    LitGraphMatchingNet,
)
from graph_mixup.mixup_generation.s_mixup.method.gmnet.training import (
    train_gmnet,
)
from graph_mixup.mixup_generation.s_mixup.method.mixup import (
    batched_mixup,
)

logger = logging.getLogger(__name__)


class InsufficientMixupItemsException(Exception): ...


def _corrupt_label(graph: Data, p: float) -> Data:
    """Corrupt the label of a graph with probability p."""
    assert type(graph.y) is torch.Tensor, "Label must be a Tensor."
    assert (
        graph.y.dim() == 2 and graph.y.size(0) == 1
    ), "Label must be one-hot encoded."

    if np.random.rand() < p:
        original_label = int(graph.y.squeeze().argmax().item())
        logger.debug(f"Corrupting label {original_label} of graph.")

        possible_labels = list(set(range(graph.y.size(1))) - {original_label})
        logger.debug(f"Possible new labels: {possible_labels}")

        new_label = np.random.choice(possible_labels)
        logger.debug(f"New label: {new_label}")

        graph.y = torch.zeros_like(graph.y)
        graph.y[0, new_label] = 1.0  # type: ignore

    return graph


def _compute_mixup_label(
    mixup_lambda: float,
    parent_0_label: Tensor,
    parent_1_label: Tensor,
    method_name: PreBatchMixupName,
) -> Tensor:
    if method_name in [
        PreBatchMixupName.G_MIXUP,
        PreBatchMixupName.IF_MIXUP,
        PreBatchMixupName.IF_MIXUP_SE,
        PreBatchMixupName.S_MIXUP,
        PreBatchMixupName.S_MIXUP_SE,
        PreBatchMixupName.SUBMIX,
    ]:
        return (
            mixup_lambda * parent_0_label + (1 - mixup_lambda) * parent_1_label
        )
    elif method_name in [
        PreBatchMixupName.FGW_MIXUP,
        PreBatchMixupName.GEOMIX,
        PreBatchMixupName.GEOMIX_SE,
        PreBatchMixupName.GED_MIXUP,
    ]:
        return (
            1 - mixup_lambda
        ) * parent_0_label + mixup_lambda * parent_1_label

    raise ValueError()


class RDBDataModule(AbstractDataModule, Generic[DataModuleConfigType]):
    def __init__(
        self,
        config: DataModuleConfigType,
        method_name: PreBatchMixupName | None,
        inner_fold: int | None,
        keep_sampled_mixup_graph_ids: bool,
    ) -> None:
        # Setup parameters.
        self.dataset_name = config.dataset_name
        self.data_dir = config.data_dir
        self.random_state = config.random_state
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        # Cross validation parameters.
        self.num_outer_folds = config.num_outer_folds
        self.fold = config.fold
        self.num_inner_folds = config.num_inner_folds
        self.inner_fold = inner_fold
        self.keep_sampled_mixup_graph_ids = keep_sampled_mixup_graph_ids

        # Label corruption probability.
        self.label_corruption_prob = config.label_corruption_prob

        # Mixup (prev: method + dataset) parameters.
        self.method_config = config.dataset_config.method_config
        self.method_name = method_name
        self.use_vanilla = (
            config.dataset_config.use_vanilla
            if isinstance(config.dataset_config, AugDatasetConfig)
            else True
        )
        self.augmented_ratio = (
            config.dataset_config.augmented_ratio
            if isinstance(config.dataset_config, AugDatasetConfig)
            else 0.0
        )
        logger.info(
            f"use_vanilla={self.use_vanilla}, augmented_ratio={self.augmented_ratio}"
        )

        # ===
        # Increases every time the train loader is initialized. Used to sample
        # different mixup graphs every time by adding the current count to the
        # random_state.
        # ===
        self.reload_train_loader_count = 0

        # Assigned in setup.
        self.train_set: list[Data] | None = None
        self.train_set_dataset_indices: list[int] | None = None
        self.vanilla_train_set: list[Data] | None = None
        self.all_mixup_graphs: dict[int, Data] | None = None
        self.val_set: list[Data] | None = None
        self.test_set: list[Data] | None = None

        # May be assigned in train_dataloader (if keep_sampled_mixup_graph_ids
        # is True).
        self.sampled_mixup_graph_ids: list[int] | None = None
        self.sampled_smixup_items: list[SMixupItem] | None = None

        # Method-specific: G-Mixup.
        self.graphons_features: dict[int, dict[str, Tensor]] | None = None
        self.class_labels: list[int] | None = None

        # Method-specific: S-Mixup.
        self.gmnet: LitGraphMatchingNet | None = None

        # Initialize parent class.
        db_manager = BaseHandler()
        self.vanilla_graphs = db_manager.get_vanilla_graphs(self.dataset_name)
        dataset = db_manager.get_dataset(self.dataset_name)
        super().__init__(
            config,
            dataset.num_features,
            dataset.num_classes,
            len(self.vanilla_graphs),
            config.eval_mode,
        )

    @override
    def setup(self, stage: str) -> None:
        # ===
        # Outer Cross-Validation. Split vanilla graphs into training/validation
        # and test data.
        # ===

        dataset_idx = np.arange(len(self.vanilla_graphs))
        dataset_labels = (
            np.array(
                [graph.label for graph in self.vanilla_graphs]
            )  # [num_graphs, 1, num_classes]
            .squeeze(1)  # [num_graphs, num_classes]
            .argmax(1)  # [num_graphs, ]
        )

        skf_outer = StratifiedKFold(
            self.num_outer_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        outer_cv_splits = list(skf_outer.split(dataset_idx, dataset_labels))
        outer_cv_train_idx, outer_cv_test_idx = outer_cv_splits[self.fold]

        if stage == "fit":
            if not self.eval_mode:
                # ===
                # Inner Cross-Validation: Split training data into training and
                # validation.
                # ===

                skf_inner = StratifiedKFold(
                    self.num_inner_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
                # Split.
                inner_cv_splits = list(
                    skf_inner.split(
                        outer_cv_train_idx, dataset_labels[outer_cv_train_idx]
                    )
                )
                # Get relative indices.
                inner_train_rel_idx, inner_val_rel_idx = inner_cv_splits[
                    self.inner_fold
                ]
                # Map back to original dataset indices.
                inner_cv_train_idx = outer_cv_train_idx[inner_train_rel_idx]
                inner_cv_val_idx = outer_cv_train_idx[inner_val_rel_idx]

                self.log_cv_indices(
                    inner_cv_train_idx, f"inner_fold_{self.inner_fold}-train"
                )
                self.log_cv_indices(
                    inner_cv_val_idx, f"inner_fold_{self.inner_fold}-val"
                )

            else:
                # ===
                # If in eval_mode, use complete training data for training (no
                # validation data for early stopping required).
                # ===

                inner_cv_train_idx = np.array(outer_cv_train_idx)
                inner_cv_val_idx = np.array([])

                self.log_cv_indices(inner_cv_train_idx, "eval-train")
                self.log_cv_indices(inner_cv_val_idx, "eval-val")

            # ===
            # Create PyG graphs from database graphs and keep track of original
            # IDs.
            # ===
            self.train_set = [
                self.vanilla_graphs[i].get_pyg_data()
                for i in inner_cv_train_idx
            ]
            # Maps: train set index -> dataset index.
            self.train_set_dataset_indices = [
                self.vanilla_graphs[i].index for i in inner_cv_train_idx
            ]

            # Training label corruption:
            # Flip labels with probability `self.label_corruption_prob`.
            train_flips = 0  # For logging only.
            if self.label_corruption_prob > 0.0:
                logger.info("Applying label corruption on train set ...")
                np.random.seed(self.random_state)
                for graph in self.train_set:
                    # For logging only.
                    original_label = int(graph.y.argmax().item())
                    logger.debug(f"Original label: {original_label}")

                    _corrupt_label(graph, self.label_corruption_prob)

                    # For logging only.
                    new_label = int(graph.y.argmax().item())
                    logger.debug(f"New label: {new_label}")
                    if original_label != new_label:
                        train_flips += 1

                # Log label corruption results.
                logger.info(
                    f"Label corruption on train set applied. Total flips: {train_flips} out of {len(self.train_set)}"
                )

                # Create a map: graph_id -> (possibly) corrupted label
                #                    int -> Tensor
                self.corrupted_labels_map: dict[int, Tensor] = {
                    self.vanilla_graphs[i].graph_id: self.train_set[idx].y
                    for idx, i in enumerate(inner_cv_train_idx)
                }  # type: ignore

            else:
                logger.info("No label corruption applied.")

            self.val_set = [
                self.vanilla_graphs[i].get_pyg_data() for i in inner_cv_val_idx
            ]

            # Apply label corruption on validation set (for monitoring purposes).
            val_flips = 0  # For logging only.
            if self.label_corruption_prob > 0.0:
                logger.info("Applying label corruption on val set ...")
                np.random.seed(self.random_state)
                for graph in self.val_set:
                    # For logging only.
                    original_label = int(graph.y.argmax().item())
                    logger.debug(f"Original label: {original_label}")

                    _corrupt_label(graph, self.label_corruption_prob)

                    # For logging only.
                    new_label = int(graph.y.argmax().item())
                    logger.debug(f"New label: {new_label}")
                    if original_label != new_label:
                        val_flips += 1

                # Log label corruption results.
                logger.info(
                    f"Label corruption on val set applied. Total flips: {val_flips} out of {len(self.val_set)}"
                )

            if self.method_name is None:
                return

            elif self.method_name in [
                PreBatchMixupName.FGW_MIXUP,
                PreBatchMixupName.IF_MIXUP,
                PreBatchMixupName.IF_MIXUP_SE,
                PreBatchMixupName.GED_MIXUP,
                PreBatchMixupName.GEOMIX,
                PreBatchMixupName.GEOMIX_SE,
                PreBatchMixupName.SUBMIX,
            ]:
                # ===
                # Mixup methods that store graphs in the database.
                #
                # Fetch all mixup graphs that belong to parents contained in
                # the train set.
                # ===

                vanilla_train_set_graph_ids = [
                    self.vanilla_graphs[i].graph_id for i in inner_cv_train_idx
                ]

                db_mixup_graphs = MixupGraphFetcher(
                    self.dataset_name,
                    self.method_name,
                    self.method_config,
                    vanilla_train_set_graph_ids,
                    self.config.dataset_config.ged_filter_flags,
                ).fetch_mixup_graphs()

                # ===
                # Check if enough mixup graphs are available.
                # ===

                num_results = len(db_mixup_graphs)
                num_required = round(len(self.train_set) * self.augmented_ratio)
                if num_results < num_required:
                    raise InsufficientMixupItemsException(
                        f"available: {num_results}, required: {num_required}"
                    )
                logger.info(
                    f"Obtained mixup graphs. Required={num_required}, available={num_results}"
                )

                # ===
                # Convert to PyG graphs and store in state.
                # ===

                self.all_mixup_graphs = {
                    mixup_graph.graph_id: mixup_graph.get_pyg_data()
                    for mixup_graph in db_mixup_graphs
                }

                # ===
                # Update labels of mixup graphs if label corruption was applied.
                # ===
                if self.label_corruption_prob > 0.0:
                    for mixup_graph in db_mixup_graphs:
                        assert mixup_graph.mixup_attrs is not None

                        # Get (possibly) corrupted parent labels.
                        parent_0_id = int(mixup_graph.mixup_attrs.parent_0_id)  # type: ignore
                        parent_1_id = int(mixup_graph.mixup_attrs.parent_1_id)  # type: ignore
                        parent_0_label: Tensor = self.corrupted_labels_map[
                            parent_0_id
                        ]
                        parent_1_label: Tensor = self.corrupted_labels_map[
                            parent_1_id
                        ]

                        # Log.
                        logger.debug(
                            f"Mixup graph {mixup_graph.graph_id} parents: "
                            f"{parent_0_id} ({parent_0_label}), "
                            f"{parent_1_id} ({parent_1_label})"
                        )

                        # Compute new label as average of (possibly) corrupted
                        # parent labels.
                        self.all_mixup_graphs[
                            mixup_graph.graph_id
                        ].y = _compute_mixup_label(
                            mixup_graph.mixup_attrs.mixup_lambda,
                            parent_0_label,
                            parent_1_label,
                            self.method_name,
                        )

                        # Log.
                        logger.debug(
                            f"Mixup graph {mixup_graph.graph_id} label: "
                            f"{self.all_mixup_graphs[mixup_graph.graph_id].y}"
                            f" (lambda={mixup_graph.mixup_attrs.mixup_lambda})"
                            f" Original label: {mixup_graph.label}"
                        )

            elif self.method_name is PreBatchMixupName.G_MIXUP:
                # ===
                # G-Mixup: Mixup graphs are generated "on-the-fly" at train
                # loader initialization.
                #
                # Graphons are computed here and are then cached.
                # ===

                logger.info("G-Mixup: Compute graphons ...")
                self.graphons_features = compute_class_graphons_and_features(
                    self.train_set
                )
                self.class_labels = list(self.graphons_features.keys())
                logger.info("G-Mixup: Graphons computed.")

            elif self.method_name is PreBatchMixupName.S_MIXUP:
                # ===
                # S-Mixup: Mixup graphs are generated "on-the-fly" at train
                # loader initialization.
                #
                # Train GMNet and store in state.
                # ===
                logger.info("S-Mixup: Train GMNet ...")
                self.gmnet = train_gmnet(
                    train_set=self.train_set,
                    node_feat_dim=self.num_features,
                    training_config=self.config.dataset_config.lit_gmnet_training_config,
                    lit_gmnet_config=self.config.dataset_config.lit_gmnet_config,
                    num_workers=self.config.num_workers,
                    device=self.config.device,
                )
                logger.info("S-Mixup: GMNet training completed.")

            else:
                assert_never(self.method_name)

        elif stage == "test":
            self.log_cv_indices(outer_cv_test_idx, "test")
            self.test_set = [
                self.vanilla_graphs[i].get_pyg_data() for i in outer_cv_test_idx
            ]

    @override
    def train_dataloader(self) -> DataLoader:
        if self.method_name is None:
            return super().train_dataloader()

        # ===
        # All mixup methods: Mixup graphs are added to the training data.
        #
        # Before sampling new mixup graphs, reset train set to vanilla
        # graphs.
        # ===

        if self.vanilla_train_set is None:
            self.vanilla_train_set = self.train_set
        self.train_set = self.vanilla_train_set

        # ===
        # Sample mixup graphs: Compute a new random sample every time
        # the dataloader is reloaded.
        # ===

        num_mixup_graphs = round(len(self.train_set) * self.augmented_ratio)
        random.seed(self.random_state + self.reload_train_loader_count)
        self.reload_train_loader_count += 1

        if self.method_name in [
            PreBatchMixupName.FGW_MIXUP,
            PreBatchMixupName.IF_MIXUP,
            PreBatchMixupName.IF_MIXUP_SE,
            PreBatchMixupName.GED_MIXUP,
            PreBatchMixupName.GEOMIX,
            PreBatchMixupName.GEOMIX_SE,
            PreBatchMixupName.SUBMIX,
        ]:
            # ===
            # Mixup methods that store graphs in the database.
            #
            # Sample mixup graphs from all available ones (fetched in `setup`).
            # ===

            sampled_mixup_graph_ids = random.sample(
                sorted(self.all_mixup_graphs.keys()),
                num_mixup_graphs,
            )
            sampled_mixup_graphs = [
                self.all_mixup_graphs[i] for i in sampled_mixup_graph_ids
            ]
            logger.debug(
                f"Sampled {len(sampled_mixup_graphs)} mixup graphs, iteration={self.reload_train_loader_count}"
            )
            if self.keep_sampled_mixup_graph_ids:
                self.sampled_mixup_graph_ids = sampled_mixup_graph_ids

        elif self.method_name is PreBatchMixupName.G_MIXUP:
            # ===
            # G-Mixup: Mixup graphs are generated "on-the-fly" below.
            # ===

            sampled_mixup_graphs: list[Data] = []
            for _ in range(num_mixup_graphs):
                class_indices = random.sample(self.class_labels, 2)
                mixup_graph = sample_graph(
                    (
                        self.graphons_features[class_indices[0]],
                        self.graphons_features[class_indices[1]],
                    ),
                    self.method_config,
                )
                sampled_mixup_graphs.append(mixup_graph)
            logger.debug(
                f"Created {len(sampled_mixup_graphs)} graphs with G-Mixup."
            )

        elif self.method_name is PreBatchMixupName.S_MIXUP:
            # ===
            # S-Mixup: Mixup graphs are generated "on-the-fly" below.
            # ===

            sampled_smixup_items: list[SMixupItem] = []
            while len(sampled_smixup_items) < num_mixup_graphs:
                # Determine the no. of mixup graphs to compute in this iteration.
                missing_items = num_mixup_graphs - len(sampled_smixup_items)
                num_items = min(128, missing_items)

                # Sample indices, graphs, and lambdas.
                indices0 = np.random.choice(len(self.train_set), num_items)
                indices1 = np.random.choice(len(self.train_set), num_items)
                graphs0 = [self.train_set[i] for i in indices0]
                graphs1 = [self.train_set[j] for j in indices1]
                lambdas = np.random.beta(
                    self.method_config.mixup_alpha,
                    self.method_config.mixup_alpha,
                    num_items,
                )

                # Compute mixup graphs.
                start_time = time.perf_counter()
                mixup_graphs = batched_mixup(
                    Batch.from_data_list(graphs0),
                    Batch.from_data_list(graphs1),
                    lambdas=lambdas,
                    graph_matching_network=self.gmnet,
                    config=self.method_config,
                    device=self.config.device,
                )
                end_time = time.perf_counter()

                # Store computed mixup graphs.
                sampled_smixup_items.extend(
                    [
                        SMixupItem(
                            graph=mixup_graph,
                            lam=float(lam),
                            source_indices=(
                                self.train_set_dataset_indices[int(idx0)],
                                self.train_set_dataset_indices[int(idx1)],
                            ),
                            creation_time_us=int(
                                (end_time - start_time) * 1e6 / num_items
                            ),
                        )
                        for (idx0, idx1, mixup_graph, lam) in zip(
                            indices0, indices1, mixup_graphs, lambdas
                        )
                    ]
                )
                logger.debug(
                    f"Graphs created ... (no. = {len(sampled_smixup_items)})"
                )

            logger.debug(
                f"Created {len(sampled_smixup_items)} graphs with S-Mixup."
            )

            sampled_mixup_graphs = [item.graph for item in sampled_smixup_items]

            if self.keep_sampled_mixup_graph_ids:
                # ===
                #  Keep items to store in test-time graph database.
                # ===
                self.sampled_smixup_items = sampled_smixup_items

        else:
            assert_never(self.method_name)

        # ===
        # All mixup methods: Combine vanilla and mixup graphs (if requested).
        # ===

        self.train_set = (
            self.train_set + sampled_mixup_graphs
            if self.use_vanilla
            else sampled_mixup_graphs
        )

        # ===
        # GeoMix, If-Mixup, & S-Mixup: Add edge weights to vanilla items.
        # ===
        if self.method_name in [
            PreBatchMixupName.IF_MIXUP,
            PreBatchMixupName.S_MIXUP,
            PreBatchMixupName.GEOMIX,
        ]:
            for graph in self.train_set:
                if graph.edge_weight is None:
                    graph.edge_weight = torch.ones(
                        graph.num_edges, dtype=torch.float
                    )

        return super().train_dataloader()
