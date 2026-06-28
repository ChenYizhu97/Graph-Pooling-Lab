from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.nn.resolver import activation_resolver

from ..functional import topk
from .contracts import PoolOutput


class SparsePooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        act: Union[str, Callable] = "tanh",
    ) -> None:
        super().__init__()
        self.select = SelectSparse(in_channels, ratio=ratio, act=act)
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> PoolOutput:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        selection = self.select(x, batch)
        scores = selection.weight
        if scores is None:
            raise RuntimeError("SparsePooling selection did not return scores.")
        perm = selection.node_index
        pooled_x = x[perm] * scores.unsqueeze(-1)
        connected = self.connect(selection, edge_index, edge_attr, batch)

        return PoolOutput(
            x=pooled_x,
            edge_index=connected.edge_index,
            batch=connected.batch,
            edge_attr=connected.edge_attr,
            perm=perm,
            score=scores,
        )


class SelectSparse(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        act: Union[str, Callable] = "tanh",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.linear = Linear(in_channels, 1)
        self.activation = activation_resolver(act)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor) -> SelectOutput:
        scores = self.activation(self.linear(x).squeeze(-1))
        node_index = topk(scores, ratio=self.ratio, batch=batch)
        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=scores[node_index],
        )
