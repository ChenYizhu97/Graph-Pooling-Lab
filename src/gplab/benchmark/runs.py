"""Resolve requested seed policies into individual, replayable dataset partitions."""
from dataclasses import dataclass

from gplab.data.dataset import build_split_indices

from .config import TrainingConfig
from .seeds import resolve_seeds


@dataclass(frozen=True)
class SplitIndices:
    """Ordered dataset indices for training, validation, and final testing."""
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: dict) -> "SplitIndices":
        return cls(**{name: tuple(value[name]) for name in ("train", "val", "test")})

    def to_mapping(self) -> dict:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    def validate(self, dataset_size: int) -> None:
        """Require nonempty, disjoint partitions covering the loaded dataset exactly once."""
        if not self.train or not self.val or not self.test:
            raise ValueError("Every run must have nonempty train, validation, and test splits.")
        indices = (*self.train, *self.val, *self.test)
        if any(type(index) is not int or not 0 <= index < dataset_size for index in indices):
            raise ValueError("Run split contains an invalid or out-of-range dataset index.")
        if len(set(indices)) != len(indices):
            raise ValueError("Run split indices must not repeat within or across partitions.")
        if len(indices) != dataset_size:
            raise ValueError("Run splits must cover the entire dataset.")


@dataclass(frozen=True)
class RunSpec:
    """One training repetition: its random seed and exact dataset partition."""
    seed: int
    split: SplitIndices

    @classmethod
    def from_mapping(cls, value: dict) -> "RunSpec":
        return cls(seed=value["seed"], split=SplitIndices.from_mapping(value["split"]))

    def to_mapping(self) -> dict:
        return {"seed": self.seed, "split": self.split.to_mapping()}


def resolve_runs(
    training: TrainingConfig,
    dataset_size: int,
    fixed_runs: tuple[RunSpec, ...] | None = None,
) -> tuple[RunSpec, ...]:
    """Use recorded runs verbatim for replay; otherwise resolve the configured seed policy.

    Both paths validate against the loaded dataset before any model is trained.
    Fixed runs override seed/split generation without rewriting the requested config.
    """
    if fixed_runs is None:
        policy = training.seeds
        seeds = resolve_seeds(
            num_runs=training.num_runs, seed_mode=policy.mode, seed_base=policy.base,
            seed_values=None if policy.values is None else list(policy.values),
            allow_duplicate_seeds=policy.allow_duplicates,
        )
        fixed_runs = tuple(
            RunSpec(seed, SplitIndices.from_mapping(build_split_indices(
                dataset_size, seed=seed,
                split_train=training.split.train, split_val=training.split.val,
            )))
            for seed in seeds
        )
    if len(fixed_runs) != training.num_runs:
        raise ValueError("Run count must equal experiment.training.num_runs.")
    for run in fixed_runs:
        if type(run.seed) is not int or not 0 <= run.seed < 2**32:
            raise ValueError("Run seed must be an integer in [0, 2**32).")
        run.split.validate(dataset_size)
    return fixed_runs
