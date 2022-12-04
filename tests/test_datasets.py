import pytest
from src.datasets import get_dataset
from torch_geometric.data import Dataset
from typing import Tuple

@pytest.mark.parametrize(
    "name,return_value",
    [
        ("MNISTSuperpixels", Tuple[Dataset, Dataset, Dataset]),
        ("MNIST", Tuple[Dataset, Dataset, Dataset]),
        ("CIFAR10", Tuple[Dataset, Dataset, Dataset]),
        ("random", ValueError),
    ]
)
def test_mnist_superpixels(name, return_value):
    '''
    Tests that the get_dataset function returns valid values
    '''
    if isinstance(return_value, ValueError):
        with pytest.raises(return_value):
            _,_,_ = get_dataset(name)
    else:
        dataset  = get_dataset(name)
        assert isinstance(dataset, return_value)