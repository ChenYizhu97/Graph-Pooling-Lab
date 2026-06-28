from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    classification_loss: float
    auxiliary_loss: float


def _classification_and_auxiliary_loss(
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    output: Tensor,
    target: Tensor,
    auxiliary_loss: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    classification_loss = loss_fn(output, target)
    auxiliary = output.new_zeros(()) if auxiliary_loss is None else auxiliary_loss
    return classification_loss, auxiliary


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    sample_count = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, auxiliary_loss = model(data)
        classification_loss, auxiliary = _classification_and_auxiliary_loss(
            loss_fn,
            output,
            data.y,
            auxiliary_loss,
        )
        loss = classification_loss + auxiliary
        loss.backward()
        optimizer.step()

        batch_size = int(data.y.numel())
        weighted_loss += float(loss.detach()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader produced no samples.")
    return weighted_loss / sample_count


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    weighted_classification_loss = 0.0
    weighted_auxiliary_loss = 0.0
    correct = 0
    sample_count = 0

    for data in loader:
        data = data.to(device)
        output, auxiliary_loss = model(data)
        classification_loss, auxiliary = _classification_and_auxiliary_loss(
            loss_fn,
            output,
            data.y,
            auxiliary_loss,
        )

        batch_size = int(data.y.numel())
        weighted_classification_loss += float(classification_loss) * batch_size
        weighted_auxiliary_loss += float(auxiliary) * batch_size
        correct += int((output.argmax(dim=-1) == data.y).sum())
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Evaluation loader produced no samples.")
    return EvaluationResult(
        accuracy=correct / sample_count,
        classification_loss=weighted_classification_loss / sample_count,
        auxiliary_loss=weighted_auxiliary_loss / sample_count,
    )
