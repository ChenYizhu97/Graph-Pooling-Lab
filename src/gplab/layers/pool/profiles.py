"""Built-in pooling profiles and their construction paths."""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import Optional

import torch
from tgp.poolers import SAGPooling, TopkPooling
from torch.nn import Linear
from torch_geometric.nn import DenseGCNConv, GCNConv

from gplab.graph import ConnectivityType

from .dense_pool_adapter import DensePoolAdapter
from .pyg_adapters import ASAPoolAdapter
from .sparse_pool import SparsePooling
from .tgp_connect import SelectionConnect

PoolBuilder = Callable[
    [int, float, Optional[float], str | Callable],
    Optional[torch.nn.Module],
]


@dataclass(frozen=True)
class PoolingSignature:
    """One declared input-to-output connectivity transformation."""
    input_type: ConnectivityType
    output_type: ConnectivityType

    def __post_init__(self) -> None:
        if not all(isinstance(value, ConnectivityType) for value in (self.input_type, self.output_type)):
            raise TypeError("PoolingSignature domains must be ConnectivityType values.")


@dataclass(frozen=True)
class PoolingProfile:
    """Pool constructor and a non-empty relation of valid input/output graph types.

    Signatures are alternatives, not a Cartesian product of independent input
    and output sets. Several outputs may be declared for the same input; all
    must be supported downstream. Tuple declarations remain accepted for plugins.
    """
    builder: PoolBuilder
    signatures: Collection[PoolingSignature]

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("PoolingProfile.builder must be callable.")
        if not self.signatures:
            raise ValueError("PoolingProfile.signatures must be non-empty.")
        if any(not isinstance(signature, PoolingSignature) for signature in self.signatures):
            raise TypeError("PoolingProfile.signatures must contain PoolingSignature values.")
        object.__setattr__(self, "signatures", frozenset(self.signatures))

    def build(
        self,
        *,
        in_channels: int,
        ratio: float,
        avg_node_num: Optional[float],
        nonlinearity: str | Callable,
    ) -> Optional[torch.nn.Module]:
        pool = self.builder(in_channels, ratio, avg_node_num, nonlinearity)
        if pool is not None and not isinstance(pool, torch.nn.Module):
            raise TypeError(
                "PoolingProfile.builder must return torch.nn.Module or None, "
                f"got {type(pool).__name__}."
            )
        return pool

    @property
    def input_types(self) -> frozenset[ConnectivityType]:
        """Input domains explicitly declared by this implementation's profile."""
        return frozenset(signature.input_type for signature in self.signatures)

    def output_types_for(self, input_type: ConnectivityType) -> frozenset[ConnectivityType]:
        """Return all outputs for this input; an empty set means unsupported input.

        Filtering by input matters: {U -> U, W -> W} on a U dataset requires
        only U support downstream, even though the pool can also emit W elsewhere.
        """
        return frozenset(
            signature.output_type for signature in self.signatures
            if signature.input_type is input_type
        )


def _no_pool(
    _in_channels: int,
    _ratio: float,
    _avg_node_num: Optional[float],
    _nonlinearity: str | Callable,
) -> None:
    return None


def _topk_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num: Optional[float],
    nonlinearity: str | Callable,
) -> torch.nn.Module:
    pool = TopkPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity,
                       remove_self_loops=False)
    if in_channels == 1:
        # TGP treats one feature as a precomputed score by default. Standard
        # TopK still needs its normalized learned projection at width one.
        pool.selector.weight = torch.nn.Parameter(torch.empty(1, 1))
        pool.selector.reset_parameters()
    pool.connector = SelectionConnect(remove_self_loops=False)
    return pool


def _sag_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num: Optional[float],
    _nonlinearity: str | Callable,
) -> torch.nn.Module:
    # Fix the paper configuration instead of inheriting TGP's GraphConv default.
    pool = SAGPooling(in_channels, ratio=ratio, GNN=GCNConv,
                      nonlinearity="tanh", remove_self_loops=False)
    pool.connector = SelectionConnect(remove_self_loops=False)
    return pool


def _asap_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num: Optional[float],
    _nonlinearity: str | Callable,
) -> torch.nn.Module:
    return ASAPoolAdapter(in_channels, ratio)


def _sparse_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num: Optional[float],
    nonlinearity: str | Callable,
) -> torch.nn.Module:
    return SparsePooling(in_channels, ratio=ratio, act=nonlinearity)


def _dense_pool(
    pool_name: str,
    graph_assignment: bool,
    in_channels: int,
    ratio: float,
    avg_node_num: Optional[float],
    _nonlinearity: str | Callable,
) -> torch.nn.Module:
    if avg_node_num is None:
        raise ValueError("avg_node_num is required for dense pooling methods.")
    # Dense methods share one assignment width based on dataset-average size,
    # not a per-graph retained-node count. Even small graphs keep every slot.
    cluster_count = max(1, int(avg_node_num * ratio))
    assignment_layer = (
        DenseGCNConv(in_channels, cluster_count)
        if graph_assignment
        else Linear(in_channels, cluster_count)
    )
    return DensePoolAdapter(assignment_layer, pool_name)


_BINARY = ConnectivityType.BINARY
_SCALAR = ConnectivityType.SCALAR

# Declared domains follow audits/COMPARABILITY_ALIGNMENT.md. Accepting an
# edge_weight argument alone does not establish a method-faithful scalar domain.
POOLING_PROFILES: Mapping[str, PoolingProfile] = MappingProxyType({
    "nopool": PoolingProfile(
        _no_pool,
        (
            PoolingSignature(_BINARY, _BINARY),
            PoolingSignature(_SCALAR, _SCALAR),
        ),
    ),
    "topkpool": PoolingProfile(
        _topk_pool,
        (PoolingSignature(_BINARY, _BINARY),),
    ),
    "sagpool": PoolingProfile(
        _sag_pool,
        (PoolingSignature(_BINARY, _BINARY),),
    ),
    "asapool": PoolingProfile(
        _asap_pool,
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
    "sparsepool": PoolingProfile(
        _sparse_pool,
        (PoolingSignature(_BINARY, _BINARY),),
    ),
    "mincutpool": PoolingProfile(
        partial(_dense_pool, "mincutpool", False),
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
    "diffpool": PoolingProfile(
        partial(_dense_pool, "diffpool", True),
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
    "densepool": PoolingProfile(
        partial(_dense_pool, "densepool", False),
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
})

def validate_pooling_profile_name(name: str) -> bool:
    is_custom_profile = ":" in name
    if not is_custom_profile and name not in POOLING_PROFILES:
        raise ValueError(
            f"Unknown pooling method '{name}'. "
            f"Built-ins: {', '.join(POOLING_PROFILES)}"
        )
    return is_custom_profile


def load_pooling_profile(name: str) -> PoolingProfile:
    """Resolve a built-in or module:attribute profile and validate the plugin contract."""
    profile = POOLING_PROFILES.get(name)
    if profile is not None:
        return profile

    module_name, separator, profile_name = name.partition(":")
    if not separator or not module_name or not profile_name:
        raise ValueError(
            f"Unknown pooling profile '{name}'. Built-ins: {', '.join(POOLING_PROFILES)}. "
            "Custom profiles must use '<python_module>:<profile_name>'."
        )

    module = import_module(module_name)
    profile = getattr(module, profile_name, None)
    if profile is None:
        raise ValueError(
            f"Cannot find pooling profile '{profile_name}' in '{module_name}'."
        )
    if not isinstance(profile, PoolingProfile):
        raise TypeError(
            f"Custom pooling profile '{name}' must be a PoolingProfile, "
            f"got {type(profile).__name__}."
        )
    return profile
