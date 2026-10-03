"""Built-in dataset profiles and their construction paths."""

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform, Compose

from gplab.graph import ConnectivityType

DatasetBuilder = Callable[[], Dataset]


class _BinaryConnectivity(BaseTransform):
    """Keep the supplied edge topology but discard scalar connectivity values."""

    def forward(self, data):
        data.edge_weight = None
        return data


@dataclass(frozen=True)
class DatasetProfile:
    """Construct native data and expose its available connectivity representations.

    ``connectivity_type`` is the builder's native/default representation.
    Scalar data also provides binary topology by explicitly discarding weights;
    binary data cannot acquire semantic scalar values merely by adding ones.
    """
    builder: DatasetBuilder
    connectivity_type: ConnectivityType

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("DatasetProfile.builder must be callable.")
        if not isinstance(self.connectivity_type, ConnectivityType):
            raise TypeError("DatasetProfile.connectivity_type must be a ConnectivityType.")

    @property
    def connectivity_types(self) -> frozenset[ConnectivityType]:
        """Representations this profile can actually materialize."""
        return frozenset({ConnectivityType.BINARY, self.connectivity_type})

    def build(self, connectivity_type: ConnectivityType | None = None) -> Dataset:
        """Build one representation; omission preserves the native default.

        Projection runs after existing transforms on each access, including
        split views, so weighted data cannot silently reach a binary experiment.
        """
        selected = self.connectivity_type if connectivity_type is None else connectivity_type
        if not isinstance(selected, ConnectivityType):
            raise TypeError("connectivity_type must be a ConnectivityType.")
        if selected not in self.connectivity_types:
            raise ValueError(f"Dataset cannot provide {selected.value} connectivity.")
        dataset = self.builder()
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "DatasetProfile.builder must return torch_geometric.data.Dataset, "
                f"got {type(dataset).__name__}."
            )
        # Copy the dataset wrapper so a builder's cached instance retains its
        # transform and semantic metadata. PyG transforms copy individual data.
        dataset = copy(dataset)
        if selected is not self.connectivity_type:
            transforms = [] if dataset.transform is None else [dataset.transform]
            dataset.transform = Compose([*transforms, _BinaryConnectivity()])
        dataset.connectivity_type = selected
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
