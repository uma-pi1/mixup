# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import random
from collections import defaultdict
from typing import Iterable

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from graph_mixup.mixup_generation.s_mixup.utils.utils import (
    one_hot_label_decode,
)


class TripleSet(Dataset):
    r"""
    This class inherits from the :class:`torch.utils.data.Dataset` class and in
    addition to each anchor sample, it returns a random positive and negative
    sample from the dataset. A positive sample has the same label as the
    anchor sample and a negative sample has a different label than the anchor
    sample.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset for which the
            triple set will be created.
        transform (function, optional): A transformation that is applied on all
            original samples. In other words, this transformation is applied
            to the anchor, positive, and negative sample. Default is None.
    """

    def __init__(
        self, dataset: Iterable[Data], transform: None | BaseTransform = None
    ):
        self.dataset = dataset
        self.transform = transform
        self.label_to_indices = self._get_label_to_indices(dataset)
        self.labels = list(self.label_to_indices.keys())

    @staticmethod
    def _get_label_to_indices(dataset: Iterable[Data]) -> dict[int, list[int]]:
        label_to_indices: defaultdict[int, list[int]] = defaultdict(list)

        for i, data in enumerate(dataset):
            label_to_indices[one_hot_label_decode(data.y.numpy())].append(i)

        return label_to_indices

    def __getitem__(self, index):
        r"""
        For a given index, this sample returns the original/anchor sample from
        the dataset at that index and a corresponding positive, and negative
        sample.

        Args:
            index (int): The index of the anchor sample in the dataset.

        Returns:
            A tuple consisting of the anchor sample, a positive
            sample, and a negative sample respectively.
        """
        anchor_data = self.dataset[index]
        anchor_label = int(one_hot_label_decode(anchor_data.y.numpy()))

        # Sample index with same label as anchor
        pos_index = index
        while pos_index == index:
            pos_index = random.sample(self.label_to_indices[anchor_label], 1)[0]

        # Sample index with different label than anchor
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.sample(self.labels, 1)[0]
        neg_index = random.sample(self.label_to_indices[neg_label], 1)[0]

        # Obtain items
        pos_data, neg_data = self.dataset[pos_index], self.dataset[neg_index]

        # Return (transformed) items
        if self.transform is not None:
            anchor_data, pos_data, neg_data = (
                self.transform(anchor_data),
                self.transform(pos_data),
                self.transform(neg_data),
            )

        return anchor_data, pos_data, neg_data

    def __len__(self):
        r"""
        Returns:
             The number of samples in the original dataset.
        """
        return len(self.dataset)
