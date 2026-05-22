import numpy as np


def one_hot_label_decode(label: np.ndarray) -> int:
    assert isinstance(label, np.ndarray), "Label must be a numpy array"

    if len(label) == 1:
        if label.ndim == 1:
            return label.item()
        elif label.ndim == 2:
            return np.argmax(label).item()

    raise ValueError(
        f"Label must have one of the following shapes: (num_classes,) or (1, num_classes). Has shape: {label.shape}"
    )
