"""Built-in pooling profiles and their construction paths."""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from inspect import signature
from types import MappingProxyType
from typing import Optional

import torch
from tgp.poolers import SAGPooling, TopkPooling
from torch_geometric.nn import GCNConv, GraphConv

from gplab.graph import ConnectivityType

from .dense_pool import DensePooling, GraphDiffPool, MinCutPooling
from .pyg_adapters import ASAPoolAdapter
from .sparse_pool import SparsePooling
from .tgp_connect import SelectionConnect

PoolBuilder = Callable[..., Optional[torch.nn.Module]]

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
        avg_node_num: Optional[float] = None,
        **params,
    ) -> Optional[torch.nn.Module]:
        """Supply model/data context separately from method-specific constructor parameters."""
        pool = self.builder(in_channels=in_channels, avg_node_num=avg_node_num, **params)
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


def _no_pool(in_channels: int, avg_node_num=None, *, ratio=0.5) -> None:
    """Identity baseline; accepts the generic ratio option without removing nodes."""
    return None


def _selection_connector(pool, params):
    """Apply the index-order correction while retaining requested TGP connector options."""
    if params.get("degree_norm") or params.get("edge_weight_norm"):
        raise ValueError("Connectivity normalization requires a separate pooling profile with weighted output signatures.")
    pool.connector = SelectionConnect(
        reduce_op=params.get("connect_red_op", "sum"),
        remove_self_loops=params.get("remove_self_loops", False),
        degree_norm=params.get("degree_norm", False),
        edge_weight_norm=params.get("edge_weight_norm", False),
    )
    return pool


def _topk_pool(in_channels: int, avg_node_num=None, **params) -> torch.nn.Module:
    params = {"remove_self_loops": False, **params}
    pool = TopkPooling(in_channels, **params)
    if in_channels == 1:
        # TGP treats one feature as a precomputed score by default. Standard
        # TopK still needs its normalized learned projection at width one.
        pool.selector.weight = torch.nn.Parameter(torch.empty(1, 1))
        pool.selector.reset_parameters()
    return _selection_connector(pool, params)


def _sag_pool(in_channels: int, avg_node_num=None, *, GNN="GCNConv", **params) -> torch.nn.Module:
    """Use GCNConv and tanh by default; resolve the JSON scorer name before construction."""
    scorers = {"GCNConv": GCNConv, "GraphConv": GraphConv}
    if GNN not in scorers:
        raise ValueError(f"SAG GNN must be one of {tuple(scorers)}.")
    # TGP silently filters unknown GNN kwargs; fail on misspelled configuration instead.
    allowed = set(signature(SAGPooling).parameters) | set(signature(scorers[GNN]).parameters)
    unknown = params.keys() - (allowed - {"kwargs"})
    if unknown:
        raise TypeError(f"Unknown SAG parameters: {', '.join(sorted(unknown))}")
    params = {"nonlinearity": "tanh", "remove_self_loops": False, **params}
    pool = SAGPooling(in_channels, GNN=scorers[GNN], **params)
    return _selection_connector(pool, params)


def _asap_pool(in_channels: int, avg_node_num=None, **params) -> torch.nn.Module:
    return ASAPoolAdapter(in_channels, **params)


def _sparse_pool(in_channels: int, avg_node_num=None, **params) -> torch.nn.Module:
    return SparsePooling(in_channels, **params)


def _dense_pool(pool_name: str, in_channels: int,
                avg_node_num=None, *, ratio=None, k=None) -> torch.nn.Module:
    """Accept native fixed k or convert a benchmark ratio using dataset-average size."""
    if k is not None and ratio is not None:
        raise ValueError("Specify either ratio or k for dense pooling, not both.")
    if k is None:
        if avg_node_num is None:
            raise ValueError("avg_node_num is required when dense pooling uses ratio.")
        ratio = 0.5 if ratio is None else ratio
        k = int(ratio) if ratio >= 1 else max(1, int(avg_node_num * ratio))
    if type(k) is not int or k <= 0:
        raise ValueError("Dense pooling k must be a positive integer.")
    if pool_name == "diffpool":
        return GraphDiffPool(in_channels, k)
    if pool_name == "mincutpool":
        return MinCutPooling(in_channels, k, cut_loss_coeff=0.5, ortho_loss_coeff=1.0,
                             adj_transpose=False, sparse_output=True)
    return DensePooling(in_channels, k, remove_self_loops=False, degree_norm=False,
                        adj_transpose=False, sparse_output=True)


_BINARY = ConnectivityType.BINARY
_SCALAR = ConnectivityType.SCALAR

# Signatures describe method-valid transformations, not implementation executability.
# Feature-only selection can be W -> W when retained edge weights are preserved.
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
        (PoolingSignature(_BINARY, _BINARY), PoolingSignature(_SCALAR, _SCALAR)),
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
        (PoolingSignature(_BINARY, _BINARY), PoolingSignature(_SCALAR, _SCALAR)),
    ),
    "mincutpool": PoolingProfile(
        partial(_dense_pool, "mincutpool"),
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
    "diffpool": PoolingProfile(
        partial(_dense_pool, "diffpool"),
        (PoolingSignature(_BINARY, _SCALAR),),
    ),
    "densepool": PoolingProfile(
        partial(_dense_pool, "densepool"),
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
