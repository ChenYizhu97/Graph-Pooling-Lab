from importlib import import_module
from typing import Callable, Optional, Union

import torch
from torch.nn import Linear
from torch_geometric.nn import DenseGCNConv, GCNConv, GINConv, GraphConv
from torch_geometric.nn.pool import ASAPooling, TopKPooling

from gplab.utils.registry import BUILTIN_POOLS, DENSE_POOLS, SUPPORTED_CONVS
from .pool import DensePoolAdapter, PoolOutput, SAGPooling, SparsePooling


def conv_resolver(layer: str):
    if layer == "GCN":
        return GCNConv
    if layer == "GraphConv":
        return GraphConv
    if layer == "GIN":
        return lambda in_channels, out_channels: GINConv(
            nn=Linear(in_channels, out_channels)
        )
    raise ValueError(
        f"Unknown convolution layer '{layer}'. "
        f"Supported conv layers: {', '.join(SUPPORTED_CONVS)}."
    )


def _dense_cluster_size(avg_node_num: Optional[float], ratio: float) -> int:
    if avg_node_num is None:
        raise ValueError("avg_node_num is required for dense pooling methods.")
    return max(1, int(avg_node_num * ratio))


def _load_pool_factory(path: str) -> Callable[..., torch.nn.Module]:
    module_name, separator, factory_name = path.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(
            "Custom pool must use '<python_module>:<factory_name>', "
            f"got '{path}'."
        )

    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"Cannot find callable pool factory '{factory_name}' in '{module_name}'."
        )
    return factory


class _TopKPoolWrapper(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float,
        nonlinearity: Union[str, Callable],
    ) -> None:
        super().__init__()
        self.topk_pool = TopKPooling(
            in_channels,
            ratio=ratio,
            nonlinearity=nonlinearity,
        )

    def forward(self, x, edge_index, batch) -> PoolOutput:
        (
            pooled_x,
            pooled_edge_index,
            pooled_edge_attr,
            pooled_batch,
            perm,
            score,
        ) = self.topk_pool(x=x, edge_index=edge_index, batch=batch)
        return PoolOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_attr=pooled_edge_attr,
            perm=perm,
            score=score,
        )

    def reset_parameters(self) -> None:
        self.topk_pool.reset_parameters()


class _ASAPoolWrapper(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: float) -> None:
        super().__init__()
        self.asa_pool = ASAPooling(in_channels, ratio=ratio)

    def forward(self, x, edge_index, batch) -> PoolOutput:
        (
            pooled_x,
            pooled_edge_index,
            pooled_edge_attr,
            pooled_batch,
            perm,
        ) = self.asa_pool(x=x, edge_index=edge_index, batch=batch)
        return PoolOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_attr=pooled_edge_attr,
            perm=perm,
        )

    def reset_parameters(self) -> None:
        self.asa_pool.reset_parameters()


def pool_resolver(
    pool_name: Optional[str],
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num: Optional[float] = None,
    nonlinearity: Union[str, Callable] = "tanh",
) -> Optional[torch.nn.Module]:
    if pool_name in (None, "", "nopool"):
        return None
    if pool_name == "topkpool":
        return _TopKPoolWrapper(in_channels, ratio, nonlinearity)
    if pool_name == "sagpool":
        return SAGPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity)
    if pool_name == "asapool":
        return _ASAPoolWrapper(in_channels, ratio)
    if pool_name == "sparsepool":
        return SparsePooling(in_channels, ratio=ratio, act=nonlinearity)
    if pool_name in DENSE_POOLS:
        cluster_count = _dense_cluster_size(avg_node_num, ratio)
        assignment_layer = (
            DenseGCNConv(in_channels, cluster_count)
            if pool_name == "diffpool"
            else Linear(in_channels, cluster_count)
        )
        return DensePoolAdapter(assignment_layer, pool_name)
    if ":" in pool_name:
        factory = _load_pool_factory(pool_name)
        custom_pool = factory(
            in_channels=in_channels,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=nonlinearity,
        )
        if not isinstance(custom_pool, torch.nn.Module):
            raise TypeError(
                f"Custom pool factory '{pool_name}' must return torch.nn.Module, "
                f"got {type(custom_pool).__name__}."
            )
        return custom_pool

    raise ValueError(
        f"Unknown pooling method '{pool_name}'. "
        f"Built-ins: {', '.join(BUILTIN_POOLS)}. "
        "Or provide a custom factory as '<python_module>:<factory_name>'."
    )
