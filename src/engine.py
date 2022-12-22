from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.utils import calculate_accuracy


def train(
    model: nn.Module,
    iter: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.DeviceObjType,
    scheduler: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    performs one epoch of training on the model, accumulating the loss and accuracy to be returned
    """
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        data = data.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, data.y)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(outputs, data.y)
    return epoch_loss / len(iter), epoch_acc / len(iter)


def evaluate(
    model: nn.Module,
    iter: DataLoader,
    criterion: nn.Module,
    device: torch.DeviceObjType,
) -> Tuple[float, float]:
    """
    evaluates the model over the specified Validation/Test DataLoader with the given criterion.
    Loss and Accuracy are accumulated over the epoch to be returned.
    """
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        data = data.to(device)

        outputs = model(data)
        loss = criterion(outputs, data.y)

        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(outputs, data.y)
    return epoch_loss / len(iter), epoch_acc / len(iter)
