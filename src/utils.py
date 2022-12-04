from typing import Tuple

import torch
from torch import nn
from torch_geometric.utils import normalized_cut


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total parameters and trainable parameters of a given model"""
    num_parameters = 0
    num_parameters_requiring_grad = 0
    for p in model.parameters():
        numel = p.numel()
        num_parameters += numel
        if p.requires_grad:
            num_parameters_requiring_grad += numel
    return num_parameters, num_parameters_requiring_grad


def normalized_cut_2d(edge_index: torch.Tensor, pos: torch.Tensor):
    """
    Courtesy of
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_graclus.py
    """
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    pred = outputs.max(1)[1]
    return pred.eq(targets).sum().item()
