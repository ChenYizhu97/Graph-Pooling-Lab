from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Optional

from gplab.data.profiles import get_dataset_profile
from gplab.layers.conv.profiles import CONV_PROFILES
from gplab.layers.pool.profiles import validate_pooling_profile_name
from gplab.utils.validation import (
    validate_model_variant_value,
    validate_pool_ratio_value,
    validate_seed_mode_value,
)


@dataclass(frozen=True)
class ModelConfig:
    """Validated backbone widths and independent pre/post-pooling encoder choices."""
    hidden_features: int
    nonlinearity: str
    p_dropout: float
    pre_conv: str
    post_conv: str
    pre_gnn: tuple[int, ...]
    post_gnn: tuple[int, ...]
    variant: str

    def __post_init__(self) -> None:
        if self.hidden_features <= 0:
            raise ValueError("case.model.hidden_features must be positive.")
        if not self.nonlinearity:
            raise ValueError("case.model.nonlinearity must be non-empty.")
        if not 0.0 <= self.p_dropout < 1.0:
            raise ValueError("case.model.p_dropout must be in [0, 1).")
        for field, value in (("pre_conv", self.pre_conv), ("post_conv", self.post_conv)):
            if value not in CONV_PROFILES:
                raise ValueError(
                    f"Unsupported case.model.{field} '{value}'. "
                    f"Supported layers: {', '.join(CONV_PROFILES)}."
                )
        if not self.pre_gnn or any(width <= 0 for width in self.pre_gnn):
            raise ValueError("case.model.pre_gnn must be a non-empty array of positive integers.")
        if self.pre_gnn[-1] != self.hidden_features:
            raise ValueError(
                "case.model.pre_gnn must end with case.model.hidden_features "
                "so pre_conv receives the configured width."
            )
        if not self.post_gnn or any(width <= 0 for width in self.post_gnn):
            raise ValueError("case.model.post_gnn must be a non-empty array of positive integers.")
        expected_readout_width = 2 * self.hidden_features
        if self.post_gnn[0] != expected_readout_width:
            raise ValueError(
                f"case.model.post_gnn must start with {expected_readout_width}, "
                "the concatenated add/max readout width."
            )
        validate_model_variant_value(self.variant)

    @classmethod
    def from_mapping(cls, value: dict) -> ModelConfig:
        return cls(
            hidden_features=int(value["hidden_features"]),
            nonlinearity=str(value["nonlinearity"]),
            p_dropout=float(value["p_dropout"]),
            pre_conv=str(value["pre_conv"]),
            post_conv=str(value["post_conv"]),
            pre_gnn=tuple(int(width) for width in value["pre_gnn"]),
            post_gnn=tuple(int(width) for width in value["post_gnn"]),
            variant=str(value["variant"]),
        )

    def to_mapping(self) -> dict:
        value = asdict(self)
        value["pre_gnn"] = list(self.pre_gnn)
        value["post_gnn"] = list(self.post_gnn)
        return value


@dataclass(frozen=True)
class PoolConfig:
    """Pooling profile and reduction settings that contribute to benchmark identity."""
    name: str
    ratio: float
    nonlinearity: str = "tanh"

    def __post_init__(self) -> None:
        validate_pooling_profile_name(self.name)
        validate_pool_ratio_value(self.ratio)
        if not self.nonlinearity:
            raise ValueError("case.pool.nonlinearity must be non-empty.")

    @property
    def source(self) -> str:
        return "custom_profile" if ":" in self.name else "builtin"

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "ratio": self.ratio,
            "nonlinearity": self.nonlinearity,
        }


@dataclass(frozen=True)
class SplitConfig:
    """Positive train/validation fractions; the remaining fraction is the test split."""
    train: float
    val: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.train)
            or not math.isfinite(self.val)
            or self.train <= 0
            or self.val <= 0
            or self.train + self.val >= 1
        ):
            raise ValueError(
                "Invalid case.training.split. Require train > 0, val > 0, "
                "and train + val < 1."
            )

    @property
    def test(self) -> float:
        return 1.0 - self.train - self.val

    @classmethod
    def from_mapping(cls, value: dict) -> SplitConfig:
        return cls(train=float(value["train"]), val=float(value["val"]))

    def to_mapping(self, *, include_test: bool = False) -> dict:
        value = {
            "train": self.train,
            "val": self.val,
        }
        if include_test:
            value["test"] = self.test
        return value


@dataclass(frozen=True)
class SeedPolicy:
    """Choose generated or explicit run seeds, with opt-in duplicate replay."""
    mode: str
    base: int
    values: Optional[tuple[int, ...]]
    allow_duplicates: bool

    def __post_init__(self) -> None:
        validate_seed_mode_value(self.mode)
        if self.mode == "list":
            if self.values is None:
                raise ValueError("case.training.seeds.values is required when mode='list'.")
            if not self.values:
                raise ValueError("case.training.seeds.values must be non-empty.")
        elif self.values is not None:
            raise ValueError("case.training.seeds.values is only valid when mode='list'.")
        if (
            self.values is not None
            and not self.allow_duplicates
            and len(set(self.values)) != len(self.values)
        ):
            raise ValueError("Duplicate seeds require case.training.seeds.allow_duplicates=true.")

    @classmethod
    def from_mapping(cls, value: dict) -> SeedPolicy:
        raw_values = value.get("values")
        return cls(
            mode=str(value["mode"]),
            base=int(value["base"]),
            values=None if raw_values is None else tuple(int(seed) for seed in raw_values),
            allow_duplicates=bool(value["allow_duplicates"]),
        )

    def to_mapping(self) -> dict:
        return {
            "mode": self.mode,
            "base": self.base,
            "values": None if self.values is None else list(self.values),
            "allow_duplicates": self.allow_duplicates,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Training budget, optimizer settings, split fractions, and run seed policy."""
    runs: int
    lr: float
    batch_size: int
    patience: int
    epochs: int
    split: SplitConfig
    seeds: SeedPolicy

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("case.training.runs must be positive.")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("case.training.lr must be a positive finite number.")
        if self.batch_size <= 0:
            raise ValueError("case.training.batch_size must be positive.")
        if self.patience < 0:
            raise ValueError("case.training.patience must be non-negative.")
        if self.epochs <= 0:
            raise ValueError("case.training.epochs must be positive.")
        if self.seeds.mode == "list" and self.seeds.values is not None:
            if len(self.seeds.values) != self.runs:
                raise ValueError("case.training.seeds.values length must equal case.training.runs.")

    @classmethod
    def from_mapping(cls, value: dict) -> TrainingConfig:
        return cls(
            runs=int(value["runs"]),
            lr=float(value["lr"]),
            batch_size=int(value["batch_size"]),
            patience=int(value["patience"]),
            epochs=int(value["epochs"]),
            split=SplitConfig.from_mapping(value["split"]),
            seeds=SeedPolicy.from_mapping(value["seeds"]),
        )

    def to_mapping(self) -> dict:
        return {
            "runs": self.runs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "epochs": self.epochs,
            "split": self.split.to_mapping(),
            "seeds": self.seeds.to_mapping(),
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """One comparable experiment definition, excluding execution-only settings."""
    dataset: str
    pool: PoolConfig
    model: ModelConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        get_dataset_profile(self.dataset)

    @classmethod
    def from_mapping(cls, value: dict) -> BenchmarkCase:
        return cls(
            dataset=str(value["dataset"]),
            pool=PoolConfig(
                name=str(value["pool"]["name"]),
                ratio=float(value["pool"]["ratio"]),
                nonlinearity=str(value["pool"]["nonlinearity"]),
            ),
            model=ModelConfig.from_mapping(value["model"]),
            training=TrainingConfig.from_mapping(value["training"]),
        )

    @classmethod
    def from_record(cls, record: dict) -> BenchmarkCase:
        """Rebuild a replay case from the original configuration and actual run seeds.

        ``case.training.seeds`` stores the requested policy (possibly ``auto``
        with no explicit values); ``run_plan.seeds`` stores the resolved seeds.
        Replay uses those values directly instead of regenerating them from base.
        The recorded runs normally already equal len(seeds); overriding runs
        preserves the existing behavior of taking the resolved plan's run count.
        ``replace`` copies the frozen configs while retaining all other settings.
        """
        case = cls.from_mapping(record["case"])
        seeds = tuple(int(seed) for seed in record["run_plan"]["seeds"])
        return replace(
            case,
            training=replace(
                case.training,
                runs=len(seeds),
                seeds=SeedPolicy(
                    mode="list",
                    # Retain the original base as configuration metadata; list mode ignores it.
                    base=case.training.seeds.base,
                    values=seeds,
                    # Recorded duplicate runs must remain replayable without deduplication.
                    allow_duplicates=len(set(seeds)) != len(seeds),
                ),
            ),
        )

    def to_mapping(self) -> dict:
        return {
            "dataset": self.dataset,
            "pool": self.pool.to_mapping(),
            "model": self.model.to_mapping(),
            "training": self.training.to_mapping(),
        }
