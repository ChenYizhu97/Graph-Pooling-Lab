from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PoolOutput:
    x: Tensor
    edge_index: Tensor
    batch: Tensor
    edge_attr: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    perm: Optional[Tensor] = None
    score: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None


def _require_tensor(value, field: str, pool_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"Pooling '{pool_name}': {field} must be a Tensor, "
            f"got {type(value).__name__}."
        )
    return value


def _validate_optional_tensor(
    value: Optional[Tensor],
    field: str,
    pool_name: str,
    device: torch.device,
) -> None:
    if value is None:
        return
    tensor = _require_tensor(value, field, pool_name)
    if tensor.device != device:
        raise RuntimeError(f"Pooling '{pool_name}': device mismatch for {field}.")


def validate_pool_output(output, pool_name: str) -> None:
    if not isinstance(output, PoolOutput):
        raise TypeError(
            f"Pooling method '{pool_name}' must return PoolOutput, "
            f"got {type(output).__name__}."
        )

    x = _require_tensor(output.x, "x", pool_name)
    edge_index = _require_tensor(output.edge_index, "edge_index", pool_name)
    batch = _require_tensor(output.batch, "batch", pool_name)

    if x.dim() != 2:
        raise ValueError(f"Pooling '{pool_name}': x must have shape [N, F].")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Pooling '{pool_name}': edge_index must have shape [2, E].")
    if batch.dim() != 1 or batch.size(0) != x.size(0):
        raise ValueError(f"Pooling '{pool_name}': batch must have shape [N].")
    if edge_index.dtype != torch.long:
        raise TypeError(f"Pooling '{pool_name}': edge_index must use torch.long.")

    device = x.device
    if edge_index.device != device or batch.device != device:
        raise RuntimeError(f"Pooling '{pool_name}': required tensors must share one device.")

    for field in ("edge_attr", "edge_weight", "perm", "score", "aux_loss"):
        _validate_optional_tensor(getattr(output, field), field, pool_name, device)

    edge_count = edge_index.size(1)
    if output.edge_attr is not None and output.edge_attr.size(0) != edge_count:
        raise ValueError(f"Pooling '{pool_name}': edge_attr length must equal edge count.")
    if output.edge_weight is not None:
        if output.edge_weight.dim() != 1 or output.edge_weight.size(0) != edge_count:
            raise ValueError(f"Pooling '{pool_name}': edge_weight must have shape [E].")
    if output.perm is not None and output.perm.dim() != 1:
        raise ValueError(f"Pooling '{pool_name}': perm must be one-dimensional.")
    if output.score is not None:
        if output.score.dim() != 1 or output.score.size(0) != x.size(0):
            raise ValueError(f"Pooling '{pool_name}': score must have shape [N].")
    if output.aux_loss is not None and output.aux_loss.numel() != 1:
        raise ValueError(f"Pooling '{pool_name}': aux_loss must be scalar.")
