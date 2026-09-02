"""Built-in dataset profiles and their construction paths."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from gplab.graph import ConnectivityType


DatasetBuilder = Callable[[], Dataset]


@dataclass(frozen=True)
class DatasetProfile:
    builder: DatasetBuilder
    connectivity_type: ConnectivityType

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("DatasetProfile.builder must be callable.")
        if not isinstance(self.connectivity_type, ConnectivityType):
            raise TypeError("DatasetProfile.connectivity_type must be a ConnectivityType.")

    def build(self) -> Dataset:
        dataset = self.builder()
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "DatasetProfile.builder must return torch_geometric.data.Dataset, "
                f"got {type(dataset).__name__}."
            )
        dataset.connectivity_type = self.connectivity_type
        return dataset


def _load_tu_dataset(name: str) -> Dataset:
    return TUDataset(root="/tmp/TUDataset", name=name, use_node_attr=True)


_TU_DATASET_NAMES = (
    "MUTAG",
    "PROTEINS",
    "ENZYMES",
    "FRANKENSTEIN",
    "Mutagenicity",
    "AIDS",
    "DD",
    "NCI1",
    "COX2",
)


DATASET_PROFILES: Mapping[str, DatasetProfile] = MappingProxyType({
    name: DatasetProfile(
        builder=partial(_load_tu_dataset, name),
        # TU edge attributes are not scalar-connectivity declarations in GPLab.
        connectivity_type=ConnectivityType.BINARY,
    )
    for name in _TU_DATASET_NAMES
})


def get_dataset_profile(name: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{name}'. "
            f"Supported datasets: {', '.join(DATASET_PROFILES)}"
        ) from exc
