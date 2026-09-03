"""Construction and connectivity capabilities for graph convolution layers."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import nn
from torch_geometric.nn import GCNConv, GINConv, GraphConv

from gplab.graph import ConnectivityType


ConvFactory = Callable[[int, int], nn.Module]


@dataclass(frozen=True)
class ConvProfile:
    factory: ConvFactory
    connectivity_types: frozenset[ConnectivityType]

    def build(self, in_channels: int, out_channels: int) -> nn.Module:
        return self.factory(in_channels, out_channels)

    def can_consume(self, connectivity_type: ConnectivityType) -> bool:
        return connectivity_type in self.connectivity_types


def _gin_factory(in_channels: int, out_channels: int) -> nn.Module:
    return GINConv(nn=nn.Linear(in_channels, out_channels))


CONV_PROFILES: Mapping[str, ConvProfile] = MappingProxyType({
    "GCN": ConvProfile(
        GCNConv,
        frozenset({ConnectivityType.BINARY, ConnectivityType.SCALAR}),
    ),
    "GraphConv": ConvProfile(
        GraphConv,
        frozenset({ConnectivityType.BINARY, ConnectivityType.SCALAR}),
    ),
    "GIN": ConvProfile(
        _gin_factory,
        frozenset({ConnectivityType.BINARY}),
    ),
})
