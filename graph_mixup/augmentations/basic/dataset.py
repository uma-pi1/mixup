from typing import assert_never

from torch_geometric.data import Data
from typing_extensions import override

from graph_mixup.augmentations.aug_dataset import AbstractAugDataset
from graph_mixup.augmentations.basic.transforms.drop_edge import DropEdge
from graph_mixup.augmentations.basic.transforms.drop_node import DropNode
from graph_mixup.augmentations.basic.transforms.drop_path import DropPath
from graph_mixup.augmentations.basic.transforms.perturb_node_attr import (
    PerturbNodeAttr,
)
from graph_mixup.augmentations.basic.typing import (
    BasicAugDatasetMethodConfig,
    DropEdgeDatasetMethodConfig,
    DropNodeDatasetMethodConfig,
    DropPathDatasetMethodConfig,
    PerturbNodeAttrDatasetMethodConfig,
)
from graph_mixup.augmentations.typing import AugDatasetConfig


class BasicAugDataset(
    AbstractAugDataset[AugDatasetConfig[BasicAugDatasetMethodConfig]]
):
    def _transform_vanilla_item(self, item: Data) -> Data:
        return item

    @override
    def _get_aug_item(self, idx: int) -> Data:
        vanilla_item = self._sample_vanilla_item()

        if isinstance(self.method_config, DropEdgeDatasetMethodConfig):
            drop_edge_t = DropEdge(self.method_config)
            return drop_edge_t(vanilla_item)

        if isinstance(self.method_config, DropNodeDatasetMethodConfig):
            drop_node_t = DropNode(self.method_config)
            return drop_node_t(vanilla_item)

        if isinstance(self.method_config, DropPathDatasetMethodConfig):
            drop_path_t = DropPath(self.method_config)
            return drop_path_t(vanilla_item)

        if isinstance(self.method_config, PerturbNodeAttrDatasetMethodConfig):
            perturb_node_attr_t = PerturbNodeAttr(self.method_config)
            return perturb_node_attr_t(vanilla_item)

        assert_never(self.method_config)
