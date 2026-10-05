from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_max_pool


def readout(x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None) -> Tensor:
    """Concatenate graph-wise sum and max features along the channel axis."""
    pooled_add = global_add_pool(x=x, batch=batch, size=size)
    pooled_max = global_max_pool(x=x, batch=batch, size=size)
    return torch.concat((pooled_add, pooled_max), dim=-1)
