from typing import Any, Optional, Tuple

from src.utils import calculate_accuracy


def train(
    model, iter, criterion, optimizer, device, scheduler: Optional[Any] = None
) -> Tuple[float, float]:
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        targets = data.y
        if isinstance(data, list):
            data = data[0].to(device)
            targets = data[1].to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(outputs, targets)
    return epoch_loss / len(iter), epoch_acc / len(iter)


def evaluate(model, iter, criterion, device) -> Tuple[float, float]:
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    for data in iter:
        targets = data.y
        if isinstance(data, list):
            data = data[0].to(device)
            targets = data[1].to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(outputs, targets)
    return epoch_loss / len(iter), epoch_acc / len(iter)
