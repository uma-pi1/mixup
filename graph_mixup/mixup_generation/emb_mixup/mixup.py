import torch
from torch import Tensor
from torch.distributions import Beta

from graph_mixup.mixup_generation.emb_mixup.typing import (
    EmbMixupModelConfig,
)


def mixup(
    items: Tensor,
    labels: Tensor,
    config: EmbMixupModelConfig,
) -> tuple[Tensor, Tensor]:
    # Compute respective sizes.
    assert 0 <= config.augmented_ratio, "augmented_ratio must be non-negative"
    n = items.size(0)
    n_vanilla = n if config.use_vanilla else 0
    n_augmented = int(n * config.augmented_ratio)

    # Sample vanilla items.
    vanilla_indices = torch.randperm(n)[:n_vanilla]
    vanilla_items, vanilla_labels = (
        items[vanilla_indices],
        labels[vanilla_indices],
    )

    # Sample items to augment.
    augm_indices_0 = torch.randperm(n)[:n_augmented]
    augm_indices_1 = torch.randperm(n)[:n_augmented]

    # Compute mixup.
    lam = Beta(config.mixup_alpha, config.mixup_alpha).sample()
    augm_items = lam * items[augm_indices_0] + (1 - lam) * items[augm_indices_1]
    augm_labels = (
        lam * labels[augm_indices_0] + (1 - lam) * labels[augm_indices_1]
    )

    # Return stack of vanilla and augmented items.
    combined_items = torch.cat([vanilla_items, augm_items])
    combined_labels = torch.cat([vanilla_labels, augm_labels])
    return combined_items, combined_labels
