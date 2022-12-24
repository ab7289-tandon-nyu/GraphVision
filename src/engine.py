from typing import Any, Optional, Tuple

from torch_geometric.data import Data

from src.utils import calculate_accuracy


def train(
    model, iter, criterion, optimizer, device, scheduler: Optional[Any] = None
) -> Tuple[float, float]:
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        targets = None
        if isinstance(data, list):
            print(f"data[0]: {data[0]}")
            print(f"data[1]: {data[1]}")
            data: Data = data[0]
            # print(data[1])
            data.y = data[1]
            data = data.to(device)
            print(f"Data: {data}")
            # print(f"targets: {targets}")
        else:
            data = data.to(device)
            targets = data.y

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


def evaluate(model, iter, criterion, device) -> Tuple[float, float]:
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        targets = None
        if isinstance(data, list):
            data: Data = data[0]
            data.y = data[1]
            data = data.to(device)
        else:
            data = data.to(device)
            # targets = data.y

        outputs = model(data)
        loss = criterion(outputs, data.y)

        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(outputs, data.y)
    return epoch_loss / len(iter), epoch_acc / len(iter)
