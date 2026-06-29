from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

from gplab.utils.registry import SUPPORTED_CONVS
from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_variant_value,
    validate_pool_ratio_value,
    validate_pool_value,
    validate_seed_mode_value,
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int
    nonlinearity: str
    p_dropout: float
    conv_layer: str
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
        if self.conv_layer not in SUPPORTED_CONVS:
            raise ValueError(
                f"Unsupported case.model.conv_layer '{self.conv_layer}'. "
                f"Supported layers: {', '.join(SUPPORTED_CONVS)}."
            )
        if not self.pre_gnn or any(width <= 0 for width in self.pre_gnn):
            raise ValueError("case.model.pre_gnn must be a non-empty array of positive integers.")
        if self.pre_gnn[-1] != self.hidden_features:
            raise ValueError(
                "case.model.pre_gnn must end with case.model.hidden_features "
                "so conv1 receives the configured width."
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
            conv_layer=str(value["conv_layer"]),
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
    name: str
    ratio: float
    nonlinearity: str = "tanh"

    def __post_init__(self) -> None:
        validate_pool_value(self.name)
        validate_pool_ratio_value(self.ratio)
        if not self.nonlinearity:
            raise ValueError("case.pool.nonlinearity must be non-empty.")

    @property
    def source(self) -> str:
        return "custom_factory" if ":" in self.name else "builtin"

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "ratio": self.ratio,
            "nonlinearity": self.nonlinearity,
        }


@dataclass(frozen=True)
class SplitConfig:
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
    dataset: str
    pool: PoolConfig
    model: ModelConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        validate_dataset_value(self.dataset)

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
        case = cls.from_mapping(record["case"])
        seeds = tuple(int(seed) for seed in record["run_plan"]["seeds"])
        return cls(
            dataset=case.dataset,
            pool=case.pool,
            model=case.model,
            training=TrainingConfig(
                runs=len(seeds),
                lr=case.training.lr,
                batch_size=case.training.batch_size,
                patience=case.training.patience,
                epochs=case.training.epochs,
                split=case.training.split,
                seeds=SeedPolicy(
                    mode="list",
                    base=case.training.seeds.base,
                    values=seeds,
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
