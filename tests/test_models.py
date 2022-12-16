import pytest
import torch

from src.models import DeeperGCN


def test_model(test_graph_input):
    model = DeeperGCN(1, 2, 8, edge_dim=2)

    outputs = model(test_graph_input)

    assert outputs.size() == (1, 2)

def test_model_features_list(test_graph_input):
    feat_list = [2, 4, 6]

    model = DeeperGCN(1, 2, feat_list, edge_dim=2, num_layers=3)

    outputs = model(test_graph_input)

    assert outputs.size() == (1,2)


def test_model_feature_list_mismatch():
    """
    Insures that an error is raised when there is a mismatch
    between the hidden feature list and the number of layers
    """
    feat_list = [2, 4, 6]

    with pytest.raises(ValueError):
        DeeperGCN(1, 2, feat_list)
