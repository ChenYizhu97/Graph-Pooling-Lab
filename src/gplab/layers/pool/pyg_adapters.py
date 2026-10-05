"""Adapters from PyG pooling return tuples to GPLab's pool contract."""
import torch
from tgp.src import PoolingOutput
from torch import Tensor
from torch_geometric.nn.pool import ASAPooling


class ASAPoolAdapter(torch.nn.Module):
    """Expose ASAP coarsened scalar weights, starting binary graphs with unit edge weights."""
    def __init__(self, in_channels: int, ratio: float = 0.5, **params) -> None:
        super().__init__()
        self.asa_pool = ASAPooling(in_channels, ratio=ratio, **params)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> PoolingOutput:
        if edge_weight is None:
            edge_weight = x.new_ones(adj.size(1))
        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, _ = self.asa_pool(
            x=x,
            edge_index=adj,
            edge_weight=edge_weight,
            batch=batch,
        )
        # The PyG tuple exposes centers, not ASAP's soft assignment matrix;
        # leave so unset rather than inventing a hard selection.
        return PoolingOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_weight=pooled_edge_weight,
        )

    def reset_parameters(self) -> None:
        self.asa_pool.reset_parameters()
