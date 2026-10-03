from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import cumsum, scatter


def readout(x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None) -> Tensor:
    """Concatenate graph-wise sum and max features along the channel axis."""
    pooled_add = global_add_pool(x=x, batch=batch, size=size)
    pooled_max = global_max_pool(x=x, batch=batch, size=size)
    return torch.concat((pooled_add, pooled_max), dim=-1)


def dense_connect(
    x: Tensor,
    adj: Tensor,
    assignment: Tensor,
    mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    """Coarsen features and adjacency with softmax assignments, masking padded input nodes."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    if assignment.dim() == 2:
        assignment = assignment.unsqueeze(0)

    batch_size, num_nodes, _ = x.size()
    assignment = torch.softmax(assignment, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * mask
        assignment = assignment * mask

    pooled_x = torch.matmul(assignment.transpose(1, 2), x)
    pooled_adj = torch.matmul(torch.matmul(assignment.transpose(1, 2), adj), assignment)
    return pooled_x, pooled_adj


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    """Select per-graph high scores; ratio >= 1 is a count and min_score overrides ratio."""
    if min_score is not None:
        # Lower an overly high threshold below each graph's maximum score so
        # threshold selection does not discard every node in a graph.
        scores_max = scatter(x, batch, reduce="max")[batch] - tol
        scores_min = scores_max.clamp(max=min_score)
        return (x > scores_min).nonzero().view(-1)

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce="sum")

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        # Regroup graphs without destroying the descending score order within
        # each graph; stability is essential for the per-graph top-k mask below.
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters must be specified")
