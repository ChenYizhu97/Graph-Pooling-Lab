"""Adapted from PyG SAGPooling, without its extra learned selection projection."""
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from ..functional import topk
from .pooling_output import PoolingOutput


class SAGPooling(torch.nn.Module):
    """Select and gate nodes using a scalar GNN score, then keep induced edges.

    Based on the SAG papers: arxiv.org/abs/1904.08082 and arxiv.org/abs/1905.02850.

    Without ``min_score``, select ceil(ratio * N) nodes per graph (ratio >= 1
    means a fixed count) after applying ``nonlinearity``. With ``min_score``,
    use graph-wise softmax scores and a threshold, keeping at least one node.
    ``multiplier`` scales selected features. Edge weights pass through to the
    induced graph; they do not participate in attention scoring.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.gnn = GNN(in_channels, 1, **kwargs)
        self.select = SelectSAG(ratio, min_score, act=nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: OptTensor = None,
        edge_weight: OptTensor = None,
        attn: OptTensor = None,
    ) -> PoolingOutput:
        """Pool a graph batch; optional ``attn`` replaces features for scoring only."""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = self.gnn(attn, edge_index)

        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(
            select_out,
            edge_index,
            edge_weight,
            batch,
        )

        return PoolingOutput(
            x=x,
            edge_index=connect_out.edge_index,
            batch=connect_out.batch,
            edge_weight=connect_out.edge_attr,
            perm=perm,
            score=score,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')


class SelectSAG(Select):
    """Turn scalar GNN scores into selected node indices and feature gates."""

    def __init__(
            self,
            ratio,
            min_score=None,
            act="tanh",
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

    def forward(
        self,
        attn: Tensor,
        batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        """Normalize scores within each graph and select nodes without learnable weights."""
        if batch is None:
            batch = attn.new_zeros(attn.size(0), dtype=torch.long)

        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = attn.squeeze(-1)

        if self.min_score is None:
            score = self.act(attn)
        else:
            score = softmax(attn, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)
        return SelectOutput(
            node_index=node_index,
            num_nodes=score.size(0),
            cluster_index=torch.arange(node_index.size(0), device=score.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
        )
