from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_mixup.mixup_generation.s_mixup.method.gmnet.lit_graph_matching_net import (
    LitGraphMatchingNet,
)
from graph_mixup.mixup_generation.s_mixup.typing import (
    LitGMNetConfig,
    LitGMNetTrainingConfig,
)
from graph_mixup.mixup_generation.s_mixup.utils.triple_set import (
    TripleSet,
)


def train_gmnet(
    train_set: list[Data],
    node_feat_dim: int,
    training_config: LitGMNetTrainingConfig,
    lit_gmnet_config: LitGMNetConfig,
    num_workers: int,
    device: int,
    max_epochs: int = 500,
) -> LitGraphMatchingNet:
    # Initialize Graph Matching Network.
    gmnet = LitGraphMatchingNet(
        node_feat_dim=node_feat_dim, config=lit_gmnet_config
    )

    # Initialize data loader.
    triple_set = TripleSet(train_set)
    loader = DataLoader(
        triple_set,
        shuffle=True,
        batch_size=training_config.batch_size,
        num_workers=num_workers,
    )

    # Initialize trainer.
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        check_val_every_n_epoch=10,
        devices=[device],
        callbacks=[
            EarlyStopping(
                monitor="train_loss",
                patience=100,
                check_on_train_epoch_end=True,
            )
        ],
    )

    # Train and return model.
    trainer.fit(gmnet, loader)
    return gmnet
