from __future__ import annotations

from dataclasses import dataclass

from gplab.data.dataset import build_split_indices

from .case import BenchmarkCase
from .identity import compute_case_id
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
    def from_mapping(cls, value: dict) -> RunPlan:
        seeds = tuple(int(seed) for seed in value["seeds"])
        splits = tuple(SplitIndices.from_mapping(split) for split in value["splits"])
        if len(seeds) != len(splits):
            raise ValueError("Recorded run_plan seeds and splits must have the same length.")
        return cls(case_id=str(value["case_id"]), seeds=seeds, splits=splits)

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

    def validate_for_execution(self, *, runs: int, dataset_size: int) -> None:
        if len(self.splits) != runs:
            raise ValueError("Run plan length must equal case.training.runs.")
        for split in self.splits:
            indices = (*split.train, *split.val, *split.test)
            if any(index < 0 or index >= dataset_size for index in indices):
                raise ValueError("Recorded run plan contains an out-of-range dataset index.")

    def to_mapping(self) -> dict:
        return {
            "case_id": self.case_id,
            "seeds": [int(seed) for seed in self.seeds],
            "splits": [split.to_mapping() for split in self.splits],
        }
