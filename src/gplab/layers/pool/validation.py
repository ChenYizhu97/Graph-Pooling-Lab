"""Validate the sparse TGP output consumed by GPLab's graph classifier."""
from typing import Optional

import torch
from tgp.src import PoolingOutput
from torch import Tensor


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


def validate_pooling_output(output, pool_name: str) -> None:
    """Check the pool boundary shapes, tensor types, and shared device."""
    if not isinstance(output, PoolingOutput):
        raise TypeError(
            f"Pooling method '{pool_name}' must return PoolingOutput, "
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

    _validate_optional_tensor(output.edge_weight, "edge_weight", pool_name, device)

    edge_count = edge_index.size(1)
    if output.edge_weight is not None:
        if output.edge_weight.dim() != 1 or output.edge_weight.size(0) != edge_count:
            raise ValueError(f"Pooling '{pool_name}': edge_weight must have shape [E].")
    if output.loss is not None:
        if not isinstance(output.loss, dict):
            raise TypeError(f"Pooling '{pool_name}': loss must be a dictionary of weighted scalar tensors.")
        for name, value in output.loss.items():
            loss = _require_tensor(value, f"loss[{name!r}]", pool_name)
            if loss.numel() != 1 or loss.device != device:
                raise ValueError(f"Pooling '{pool_name}': loss[{name!r}] must be scalar and on {device}.")
