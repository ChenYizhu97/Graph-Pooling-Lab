import torch
from torch import Tensor


def to_sparse_batch(
    x: Tensor,
    adj: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flatten fixed cluster slots, retaining every adjacency entry including zero weights."""
    num_graphs = x.size(0)
    num_clusters = x.size(1)
    x = x.reshape((num_graphs * num_clusters, -1))

    # Dense pooling outputs fixed coarse cluster slots, so we expand the full
    # per-graph C x C adjacency and carry its values as edge weights.
    local_row = torch.arange(num_clusters, device=x.device).repeat_interleave(num_clusters)
    local_col = torch.arange(num_clusters, device=x.device).repeat(num_clusters)
    edge_template = torch.stack((local_row, local_col), dim=0)
    offsets = (torch.arange(num_graphs, device=x.device) * num_clusters).view(num_graphs, 1, 1)
    edge_index = (edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    edge_weight = adj.reshape(-1)

    batch = torch.arange(num_graphs, device=x.device).repeat_interleave(num_clusters)

    return x, edge_index, batch, edge_weight
