from typing import Tuple

import pytest
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.datasets import MNISTSuperpixels

from src.datasets import get_dataset


@pytest.mark.parametrize(
    "name,return_value",
    [
        ("MNISTSuperpixels", (Subset, Subset, MNISTSuperpixels)),
        ("MNIST", (Dataset, Dataset, Dataset)),
        ("CIFAR10", (Dataset, Dataset, Dataset)),
        ("random", "error"),
    ],
)
def test_dataset_download(name, return_value, tmp_path):
    """
    Tests that the get_dataset function returns valid values
    """

    path = tmp_path
    if isinstance(return_value, str):
        with pytest.raises(ValueError):
            _, _, _ = get_dataset(path, name)

    else:
        dataset = get_dataset(path, name)
        assert isinstance(dataset, tuple)
        assert isinstance(dataset[0], return_value[0])
        assert isinstance(dataset[1], return_value[1])
        assert isinstance(dataset[2], return_value[2])
