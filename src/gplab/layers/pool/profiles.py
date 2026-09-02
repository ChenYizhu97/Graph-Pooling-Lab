"""Built-in pooling profiles and their construction paths."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import Optional

import torch
from torch.nn import Linear
from torch_geometric.nn import DenseGCNConv

from gplab.graph import ConnectivityType

from .dense_pool_adapter import DensePoolAdapter
from .pyg_adapters import ASAPoolAdapter, TopKPoolAdapter
from .sag_pool import SAGPooling
from .sparse_pool import SparsePooling


PoolBuilder = Callable[
    [int, float, Optional[float], str | Callable],
    Optional[torch.nn.Module],
]


@dataclass(frozen=True)
class PoolingSignature:
    input_type: ConnectivityType
    output_type: ConnectivityType


@dataclass(frozen=True)
class PoolingProfile:
    builder: PoolBuilder
    signatures: tuple[PoolingSignature, ...]

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("PoolingProfile.builder must be callable.")
        if not self.signatures:
            raise ValueError("PoolingProfile.signatures must be non-empty.")
        input_types = [signature.input_type for signature in self.signatures]
        if len(input_types) != len(set(input_types)):
            raise ValueError("PoolingProfile cannot declare multiple outputs for one input type.")

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

    def output_type_for(
        self,
        input_type: ConnectivityType,
    ) -> Optional[ConnectivityType]:
        return next(
            (
                signature.output_type
                for signature in self.signatures
                if signature.input_type is input_type
            ),
            None,
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
    return TopKPoolAdapter(in_channels, ratio, nonlinearity)


def _sag_pool(
    in_channels: int,
    ratio: float,
    _avg_node_num: Optional[float],
    nonlinearity: str | Callable,
) -> torch.nn.Module:
    return SAGPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity)


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
    cluster_count = max(1, int(avg_node_num * ratio))
    assignment_layer = (
        DenseGCNConv(in_channels, cluster_count)
        if graph_assignment
        else Linear(in_channels, cluster_count)
    )
    return DensePoolAdapter(assignment_layer, pool_name)


_BINARY = ConnectivityType.BINARY
_SCALAR = ConnectivityType.SCALAR

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
