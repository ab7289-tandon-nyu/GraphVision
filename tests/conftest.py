import pytest
import torch
from torch_geometric import transforms
from torch_geometric.data import Data
from torch_geometric.datasets import MNISTSuperpixels


@pytest.fixture
def test_datasets(tmp_path):
    path = tmp_path / "data"
    path.mkdir()

    train_dataset = MNISTSuperpixels(path, True)
    test_dataset = MNISTSuperpixels(path, False)

    train_len = len(train_dataset)
    train_ratio = int(0.9 * train_len)
    valid_ratio = len(train_dataset) - train_ratio
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_ratio, valid_ratio]
    )
    return train_dataset, valid_dataset, test_dataset


@pytest.fixture
def test_graph_input():
    data = Data(
        x=torch.ones((10, 1)),
        edge_index=torch.ones((2, 15), dtype=torch.long),
        edge_attr=torch.ones((15, 2)),
        batch=torch.tensor([0]),
        y=torch.tensor([1]),
    )
    return data


@pytest.fixture
def test_transforms():
    return (
        transforms.Compose(transforms.ToSparseTensor()),
        transforms.Compose(transforms.ToSparseTensor()),
    )
