from __future__ import annotations

from dataclasses import dataclass

from gplab.data.dataset import build_split_indices
from gplab.experiment.reproducibility import resolve_seeds

from .case import BenchmarkCase
from .comparison import compute_case_id


@dataclass(frozen=True)
class RunPlan:
    case_id: str
    seeds: tuple[int, ...]
    splits: tuple[dict, ...]

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
            build_split_indices(
                dataset_size,
                seed=seed,
                split_train=training.split.train,
                split_val=training.split.val,
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
            "splits": [
                {
                    "train": [int(index) for index in split["train"]],
                    "val": [int(index) for index in split["val"]],
                    "test": [int(index) for index in split["test"]],
                }
                for split in self.splits
            ],
        }
