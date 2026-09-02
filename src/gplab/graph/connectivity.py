"""Semantic types for graph connectivity."""

from enum import Enum


class ConnectivityType(str, Enum):
    """Information carried by graph connectivity at a model boundary."""

    BINARY = "binary"
    SCALAR = "scalar"
