from typing import List, Tuple, Union

import pytest
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from src.datasets import get_dataloaders, get_datasets
from src.transforms import get_transforms_list


@pytest.mark.parametrize(
    "name,return_value",
    [
        ("MNISTSuperpixels", (Subset, Subset, MNISTSuperpixels)),
        ("MNIST", (Dataset, Dataset, Dataset)),
        ("CIFAR10", (Dataset, Dataset, Dataset)),
        ("random", "error"),
        ("torchvision_mnist", (Subset, Subset, MNIST)),
        ("torchvision_cifar10", (Subset, Subset, CIFAR10)),
    ],
)
def test_dataset_download(name, return_value, tmp_path, test_transforms):
    """
    Tests that the get_dataset function returns valid values
    """

    path = tmp_path
    if isinstance(return_value, str):
        with pytest.raises(ValueError):
            _, _, _ = get_datasets(path, name)

    else:
        dataset = get_datasets(path, name, transforms=test_transforms)
        assert isinstance(dataset, tuple)
        assert isinstance(dataset[0], return_value[0])
        assert isinstance(dataset[1], return_value[1])
        assert isinstance(dataset[2], return_value[2])


def test_create_dataloaders(test_datasets):
    """
    Tests that the create dataloader functions as expected
    """
    train_d, valid_d, test_d = test_datasets

    loaders = get_dataloaders(train_d, valid_d, test_d, batch_size=3)
    for loader in loaders:
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 3


def test_create_dataloaders_batch(test_datasets):
    """
    Tests that the different methods of specifying batch sizes work
    """
    test_batch_sizes = [
        8,
        (8,),
        (
            8,
            8,
        ),
        (8, 8, 8),
        (8, 8, 8, 8),
    ]

    for batch_size in test_batch_sizes:
        # batch_size is just int
        if isinstance(batch_size, int):
            loaders = get_dataloaders(*test_datasets, batch_size=batch_size)
            for loader in loaders:
                assert isinstance(loader, DataLoader)
                assert loader.batch_size == batch_size
        # batch size is too long
        elif len(batch_size) > 3:
            with pytest.raises(ValueError):
                _ = get_dataloaders(*test_datasets, batch_size=batch_size)
        else:
            loaders = get_dataloaders(*test_datasets, batch_size=batch_size)

            if len(batch_size) == 1:
                for loader in loaders:
                    assert isinstance(loader, DataLoader)
                    assert loader.batch_size == batch_size[0]
            elif len(batch_size) == 2:
                for idx, loader in enumerate(loaders):
                    assert isinstance(loader, DataLoader)
                    if idx <= 1:
                        assert loader.batch_size == batch_size[0]
                    else:
                        assert loader.batch_size == batch_size[1]
            else:
                for idx, loader in enumerate(loaders):
                    assert isinstance(loader, DataLoader)
                    assert loader.batch_size == batch_size[idx]


@pytest.mark.parametrize(
    "type,result_type",
    [
        ("cartesian", list),
        ("distance", list),
        ("localcartesian", list),
        ("mnist-slic", list),
        ("cifar10-slic", tuple),
    ],
)
def test_get_transform_list(type, result_type):

    result = get_transforms_list(type)
    assert isinstance(result, result_type)
