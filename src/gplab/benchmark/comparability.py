"""Assess a specified pool set under one shared dataset/model setting."""
from collections.abc import Iterable
from dataclasses import dataclass

from gplab.data.profiles import get_dataset_profile
from gplab.graph import ConnectivityType

from .compatibility import pool_compatibility_error
from .config import ModelConfig


@dataclass(frozen=True)
class ComparisonSetting:
    """Shared dataset and complete model configuration for a pool comparison.

    ``input_type=None`` searches all representations the dataset can provide;
    an explicit type restricts the comparison to that shared representation.
    This describes the structural setting; callers must also hold dataset
    instances, splits, and training rules fixed when executing the experiments.
    """
    dataset: str
    model: ModelConfig
    input_type: ConnectivityType | None = None

    def __post_init__(self) -> None:
        if self.input_type is not None and not isinstance(self.input_type, ConnectivityType):
            raise TypeError("input_type must be a ConnectivityType or None.")


@dataclass(frozen=True)
class ComparabilityResult:
    """Shared valid inputs and rejected alternatives for this pool set/setting.

    Every pool must use the same member of ``input_types``. Incompatibilities
    are grouped by rejected input type, then pool; they can be nonempty even
    when another input makes the comparison valid.
    """
    pools: tuple[str, ...]
    setting: ComparisonSetting
    input_types: frozenset[ConnectivityType]
    incompatibilities: dict[ConnectivityType, dict[str, str]]

    @property
    def comparable(self) -> bool:
        return bool(self.input_types)


def check_comparability(
    pools: Iterable[str], setting: ComparisonSetting,
) -> ComparabilityResult:
    """Check all specified pools under the same setting without loading a dataset.

    A common input must be available from the dataset and accepted by every
    pool. For that input, every possible pool output must be consumable by the
    post-encoder. Individually compatible but disjoint inputs do not suffice.
    Pool-size controls are experiment-specific, not a universal condition here.
    """
    if isinstance(pools, str):
        raise TypeError("pools must be a collection of profile names, not one string.")
    names = tuple(dict.fromkeys(pools))
    if len(names) < 2:
        raise ValueError("A comparison requires at least two distinct pooling profiles.")
    dataset_types = get_dataset_profile(setting.dataset).connectivity_types
    if setting.input_type is not None:
        if setting.input_type not in dataset_types:
            raise ValueError(f"Dataset '{setting.dataset}' cannot provide {setting.input_type.value} connectivity.")
        dataset_types = frozenset({setting.input_type})
    input_types = set()
    incompatibilities = {}
    # Testing all pools against each available type computes the intersection
    # while retaining useful per-pool reasons for rejected representations.
    for dataset_type in sorted(dataset_types, key=lambda value: value.value):
        errors = {}
        for name in names:
            error = pool_compatibility_error(
                dataset_type=dataset_type,
                pool_name=name,
                pre_conv=setting.model.pre_conv,
                post_conv=setting.model.post_conv,
            )
            if error is not None:
                errors[name] = error
        if errors:
            incompatibilities[dataset_type] = errors
        else:
            input_types.add(dataset_type)
    return ComparabilityResult(names, setting, frozenset(input_types), incompatibilities)
