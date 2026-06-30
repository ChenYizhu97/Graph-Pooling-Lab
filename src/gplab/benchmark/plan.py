from __future__ import annotations

from dataclasses import dataclass

from gplab.data.dataset import build_split_indices

from .case import BenchmarkCase
from .comparison import compute_case_id
from .seeds import resolve_seeds


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: dict) -> SplitIndices:
        return cls(
            train=tuple(int(index) for index in value["train"]),
            val=tuple(int(index) for index in value["val"]),
            test=tuple(int(index) for index in value["test"]),
        )

    def to_mapping(self) -> dict:
        return {
            "train": [int(index) for index in self.train],
            "val": [int(index) for index in self.val],
            "test": [int(index) for index in self.test],
        }


@dataclass(frozen=True)
class RunPlan:
    case_id: str
    seeds: tuple[int, ...]
    splits: tuple[SplitIndices, ...]

    @classmethod
    def build(cls, case: BenchmarkCase, dataset_size: int) -> RunPlan:
        training = case.training
        seed_policy = training.seeds
        seeds = resolve_seeds(
            runs=training.runs,
            seed_mode=seed_policy.mode,
            seed_base=seed_policy.base,
            seed_values=None if seed_policy.values is None else list(seed_policy.values),
            allow_duplicate_seeds=seed_policy.allow_duplicates,
        )
        splits = [
            SplitIndices.from_mapping(
                build_split_indices(
                    dataset_size,
                    seed=seed,
                    split_train=training.split.train,
                    split_val=training.split.val,
                )
            )
            for seed in seeds
        ]
        return cls(
            case_id=compute_case_id(case),
            seeds=tuple(int(seed) for seed in seeds),
            splits=tuple(splits),
        )

    def to_mapping(self) -> dict:
        return {
            "case_id": self.case_id,
            "seeds": [int(seed) for seed in self.seeds],
            "splits": [split.to_mapping() for split in self.splits],
        }
