import pytest
import torch

from src.models import DeeperGCN


def test_model(test_graph_input):
    model = DeeperGCN(1, 2, 8, edge_dim=2)

    outputs = model(test_graph_input)

    assert outputs.size() == (1, 2)
