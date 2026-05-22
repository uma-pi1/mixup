from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch.nn.functional import one_hot


class OneHotLabel(BaseTransform):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def forward(self, data: Data) -> Data:
        data.y = one_hot(data.y, self.num_classes).float()
        return data
