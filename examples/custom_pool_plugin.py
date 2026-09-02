"""Example custom pooling profile for Graph Pooling Lab.

Usage:
    gplab-train \
        --pool examples.custom_pool_plugin:CUSTOM_POOL_PROFILE \
        --pool-ratio 0.6
"""

import torch
from torch_geometric.nn.pool import TopKPooling

from gplab.graph import ConnectivityType
from gplab.layers.pool import PoolingOutput, PoolingProfile, PoolingSignature


class CustomTopKPool(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.pool = TopKPooling(in_channels, ratio=ratio)

    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_weight=None,
    ) -> PoolingOutput:
        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, perm, score = (
            self.pool(
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
        self.pool.reset_parameters()


def _build_custom_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num,
    _nonlinearity,
) -> torch.nn.Module:
    return CustomTopKPool(in_channels, ratio=ratio)


CUSTOM_POOL_PROFILE = PoolingProfile(
    builder=_build_custom_pool,
    signatures=(
        PoolingSignature(
            ConnectivityType.BINARY,
            ConnectivityType.BINARY,
        ),
    ),
)
