import copy
from typing import Any, Tuple, Union

import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torchvision import datasets as vision_datasets

_DATASET_NAMES = dict(
    MNIST_P="MNISTSuperpixels",
    MNIST="MNIST",
    CIFAR10="CIFAR10",
    T_CIFAR10="torchvision_cifar10",
    T_MNIST="torchvision_mnist",
)


def get_datasets(
    data_dir: str = "../data/",
    name: str = "MNISTSuperpixels",
    pre_transforms: Any = None,
    transforms: Union[Any, Tuple[Any, Any]] = None,
    valid_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns a tuple of the train, validation, and test datasets of the specified
    data, with any transforms or pre-transforms applied.
    """
    if name == _DATASET_NAMES["MNIST_P"]:
        train_dataset = datasets.MNISTSuperpixels(
            data_dir, train=True, pre_transform=pre_transforms, transform=transforms
        )
        test_dataset = datasets.MNISTSuperpixels(
            data_dir, train=False, pre_transform=pre_transforms, transform=transforms
        )

        train_dataset, valid_dataset = calculate_split(
            train_dataset, valid_ratio, transforms
        )

        return train_dataset, valid_dataset, test_dataset
    elif name == _DATASET_NAMES["MNIST"] or name is _DATASET_NAMES["CIFAR10"]:
        train_dataset = datasets.GNNBenchmarkDataset(
            data_dir,
            name,
            split="train",
            pre_transform=pre_transforms,
            transform=transforms,
        )
        valid_dataset = datasets.GNNBenchmarkDataset(
            data_dir,
            name,
            split="val",
            pre_transform=pre_transforms,
            transform=transforms,
        )
        test_dataset = datasets.GNNBenchmarkDataset(
            data_dir,
            name,
            split="test",
            pre_transform=pre_transforms,
            transform=transforms,
        )
        return train_dataset, valid_dataset, test_dataset
    elif name == _DATASET_NAMES["T_MNIST"]:
        train_dataset = vision_datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms,
        )
        test_dataset = vision_datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms,
        )

        train_dataset, valid_dataset = calculate_split(
            train_dataset, valid_ratio, transforms
        )

        return train_dataset, valid_dataset, test_dataset
    elif name == _DATASET_NAMES["T_CIFAR10"]:
        if not transforms or not len(transforms) == 2:
            raise ValueError("Torchvision datasets need train and test transforms")
        train_transforms, test_transforms = transforms

        train_dataset = vision_datasets.CIFAR10(
            data_dir,
            train=True,
            transform=train_transforms,
            download=True,
        )
        test_dataset = vision_datasets.CIFAR10(
            data_dir,
            train=False,
            transform=test_transforms,
            download=True,
        )

        train_dataset, valid_dataset = calculate_split(
            train_dataset, valid_ratio, train_transforms
        )

        return train_dataset, valid_dataset, test_dataset
    else:
        raise ValueError("Invalid Dataset name")


def get_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: Union[int, Tuple[int, int]] = 1,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Given valid, train, and test datasets; returns dataloadersr with the specified batch size
    """
    train_batch, valid_batch, test_batch = None, None, None
    if isinstance(batch_size, Tuple):
        if len(batch_size) == 1:
            train_batch, valid_batch, test_batch = (
                batch_size[0],
                batch_size[0],
                batch_size[0],
            )
        elif len(batch_size) == 2:
            train_batch, valid_batch, test_batch = (
                batch_size[0],
                batch_size[0],
                batch_size[1],
            )
        elif len(batch_size) == 3:
            train_batch, valid_batch, test_batch = (
                batch_size[0],
                batch_size[1],
                batch_size[2],
            )
        else:
            raise ValueError("Invalid number of indices for batch_size")
    else:
        train_batch, valid_batch, test_batch = batch_size, batch_size, batch_size
    return (
        # Train
        DataLoader(
            train_dataset, batch_size=train_batch, shuffle=True, drop_last=drop_last
        ),
        # Valid
        DataLoader(
            valid_dataset, batch_size=valid_batch, shuffle=False, drop_last=drop_last
        ),
        # Test
        DataLoader(
            test_dataset, batch_size=test_batch, shuffle=False, drop_last=drop_last
        ),
    )


def calculate_split(dataset, valid_ratio, transform):
    """
    Calculates the Training set/Validation set split
    """
    train_len = int(len(dataset) * (1 - valid_ratio))
    valid_len = int(len(dataset) - train_len)
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])
    valid_dataset.dataset.transform = transform
    return train_dataset, copy.deepcopy(valid_dataset)
