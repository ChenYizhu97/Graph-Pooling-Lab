import torch
from torch import Tensor


def to_sparse_batch(
    x: Tensor,
    adj: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch_num = x.size(0)
    node_num = x.size(1)
    x = x.reshape((batch_num * node_num, -1))

    # Dense pooling outputs fixed coarse cluster slots, so we expand the full
    # per-graph C x C adjacency and carry its values as edge weights.
    local_row = torch.arange(node_num, device=x.device).repeat_interleave(node_num)
    local_col = torch.arange(node_num, device=x.device).repeat(node_num)
    edge_template = torch.stack((local_row, local_col), dim=0)
    offsets = (torch.arange(batch_num, device=x.device) * node_num).view(batch_num, 1, 1)
    edge_index = (edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    edge_weight = adj.reshape(-1)

    batch = torch.arange(batch_num, device=x.device).repeat_interleave(node_num)

    return x, edge_index, batch, edge_weight
