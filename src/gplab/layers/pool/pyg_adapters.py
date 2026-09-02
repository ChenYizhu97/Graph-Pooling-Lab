"""Adapters from PyG pooling return tuples to GPLab's pool contract."""
from collections.abc import Callable

import torch
from torch import Tensor
from torch_geometric.nn.pool import ASAPooling, TopKPooling

from .pooling_output import PoolingOutput


class TopKPoolAdapter(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float,
        nonlinearity: str | Callable,
    ) -> None:
        super().__init__()
        self.topk_pool = TopKPooling(
            in_channels,
            ratio=ratio,
            nonlinearity=nonlinearity,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> PoolingOutput:
        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, perm, score = (
            self.topk_pool(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                batch=batch,
            )
        )
        return PoolingOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_weight=pooled_edge_weight,
            perm=perm,
            score=score,
        )

    def reset_parameters(self) -> None:
        self.topk_pool.reset_parameters()


class ASAPoolAdapter(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: float) -> None:
        super().__init__()
        self.asa_pool = ASAPooling(in_channels, ratio=ratio)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> PoolingOutput:
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, perm = self.asa_pool(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
        )
        return PoolingOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_weight=pooled_edge_weight,
            perm=perm,
        )

    def reset_parameters(self) -> None:
        self.asa_pool.reset_parameters()
