from importlib import import_module
from typing import Callable, Optional, Union
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv, DenseGCNConv, GINConv
from torch_geometric.nn.pool import TopKPooling, ASAPooling
from .pool import DensePoolAdapter, SAGPooling, SparsePooling, PoolOutput
from gplab.utils.registry import BUILTIN_POOLS, DENSE_POOLS

SUPPORTED_CONVS = ("GCN", "GraphConv", "GIN")


def conv_resolver(layer: str):
    # Add more convolution layer resolvers here when the benchmark supports them.
    if layer == "GCN":
        return GCNConv
    if layer == "GraphConv":
        return GraphConv
    if layer == "GIN":
        return lambda in_channel, out_channel: GINConv(nn=Linear(in_channel, out_channel))
    raise ValueError(
        f"Unknown convolution layer '{layer}'. "
        f"Supported conv layers: {', '.join(SUPPORTED_CONVS)}."
    )


def _dense_cluster_size(avg_node_num: Optional[float], ratio: float) -> int:
    if avg_node_num is None:
        raise ValueError("avg_node_num is required for dense pooling methods.")
    return max(1, int(avg_node_num * ratio))


def _load_pool_factory(path: str) -> Callable[..., torch.nn.Module]:
    module_name, sep, factory_name = path.partition(":")
    if sep == "" or not module_name or not factory_name:
        raise ValueError(
            "Custom pool must use '<python_module>:<factory_name>', "
            f"got '{path}'."
        )

    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Cannot find callable pool factory '{factory_name}' in '{module_name}'.")
    return factory


class _TopKPoolWrapper(torch.nn.Module):
    """
    Wrapper for PyG TopKPooling to return PoolOutput.
    
    PyG TopKPooling returns: (x, edge_index, edge_attr, batch, perm, score)
    """
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        nonlinearity: Union[str, callable] = "tanh",
    ):
        super().__init__()
        self.pool = TopKPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity)
    
    def forward(self, x, edge_index, batch):
        x_out, edge_index_out, edge_attr_out, batch_out, perm, score = self.pool(
            x=x, edge_index=edge_index, batch=batch
        )
        return PoolOutput(
            x=x_out,
            edge_index=edge_index_out,
            batch=batch_out,
            edge_attr=edge_attr_out,
            perm=perm,
            score=score,
        )
    
    def reset_parameters(self):
        self.pool.reset_parameters()


class _ASAPoolWrapper(torch.nn.Module):
    """
    Wrapper for PyG ASAPooling to return PoolOutput.
    
    PyG ASAPooling returns: (x, edge_index, edge_attr, batch, perm)
    Note: ASAPooling does not return scores.
    """
    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.pool = ASAPooling(in_channels, ratio=ratio)
    
    def forward(self, x, edge_index, batch):
        x_out, edge_index_out, edge_attr_out, batch_out, perm = self.pool(
            x=x, edge_index=edge_index, batch=batch
        )
        return PoolOutput(
            x=x_out,
            edge_index=edge_index_out,
            batch=batch_out,
            edge_attr=edge_attr_out,
            perm=perm,
            score=None,  # ASAPooling doesn't return scores
        )
    
    def reset_parameters(self):
        self.pool.reset_parameters()


def pool_resolver(
    pool: str,
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num: Optional[float] = None,
    nonlinearity: Union[str, callable] = "relu",
) -> Optional[torch.nn.Module]:
    """
    Resolve pooling method name to pooling layer instance.
    
    All returned pooling layers are guaranteed to return PoolOutput from forward().
    
    Args:
        pool: Pooling method name or custom factory path
        in_channels: Number of input channels
        ratio: Pooling ratio (default: 0.5)
        avg_node_num: Average number of nodes per graph (required for dense pooling)
        nonlinearity: Nonlinearity to use
        
    Returns:
        Pooling layer instance or None for "nopool"
    """
    # For dense pooling methods, this returns the learnable assignment module wrapped by DensePoolAdapter.
    if pool in (None, "", "nopool"):
        return None

    if pool == "topkpool":
        return _TopKPoolWrapper(in_channels, ratio=ratio, nonlinearity=nonlinearity)
    
    if pool == "sagpool":
        # SAGPooling already returns PoolOutput
        return SAGPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity)
    
    if pool == "asapool":
        return _ASAPoolWrapper(in_channels, ratio=ratio)
    
    if pool == "sparsepool":
        # SparsePooling already returns PoolOutput
        return SparsePooling(in_channels, ratio=ratio, act=nonlinearity)

    if pool in DENSE_POOLS:
        k = _dense_cluster_size(avg_node_num, ratio)
        assignment_pool = DenseGCNConv(in_channels, k) if pool == "diffpool" else Linear(in_channels, k)
        return DensePoolAdapter(
            assignment_pool,
            pool,
            nonlinearity=nonlinearity,
        )

    if ":" in pool:
        factory = _load_pool_factory(pool)
        custom_pool = factory(
            in_channels=in_channels,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=nonlinearity,
        )

        # Custom pooling is expected to return PoolOutput directly.
        # First-batch validation will catch contract violations.
        return custom_pool

    raise ValueError(
        f"Unknown pooling method '{pool}'. "
        f"Built-ins: {', '.join(BUILTIN_POOLS)}. "
        "Or provide a custom factory as '<python_module>:<factory_name>'."
    )
