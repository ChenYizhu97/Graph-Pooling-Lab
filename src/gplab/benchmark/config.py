"""Validated experiment settings, independent of logging and resolved runs."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Optional

from gplab.benchmark.compression import CompressionControl
from gplab.data.profiles import get_dataset_profile
from gplab.graph import ConnectivityType
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
            raise ValueError("experiment.model.hidden_features must be positive.")
        if not self.nonlinearity:
            raise ValueError("experiment.model.nonlinearity must be non-empty.")
        if not 0.0 <= self.p_dropout < 1.0:
            raise ValueError("experiment.model.p_dropout must be in [0, 1).")
        for field_name, value in (("pre_conv", self.pre_conv), ("post_conv", self.post_conv)):
            if value not in CONV_PROFILES:
                raise ValueError(
                    f"Unsupported experiment.model.{field_name} '{value}'. "
                    f"Supported layers: {', '.join(CONV_PROFILES)}."
                )
        if not self.pre_gnn or any(width <= 0 for width in self.pre_gnn):
            raise ValueError("experiment.model.pre_gnn must be a non-empty array of positive integers.")
        if self.pre_gnn[-1] != self.hidden_features:
            raise ValueError(
                "experiment.model.pre_gnn must end with experiment.model.hidden_features "
                "so pre_conv receives the configured width."
            )
        if not self.post_gnn or any(width <= 0 for width in self.post_gnn):
            raise ValueError("experiment.model.post_gnn must be a non-empty array of positive integers.")
        expected_readout_width = 2 * self.hidden_features
        if self.post_gnn[0] != expected_readout_width:
            raise ValueError(
                f"experiment.model.post_gnn must start with {expected_readout_width}, "
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
    """Method name and native JSON constructor parameters, independent of comparability."""
    name: str
    params: dict = field(default_factory=lambda: {"ratio": 0.5})

    def __post_init__(self) -> None:
        validate_pooling_profile_name(self.name)
        if not isinstance(self.params, dict):
            raise ValueError("experiment.pool.params must be an object.")
        if "in_channels" in self.params or "avg_node_num" in self.params:
            raise ValueError("experiment.pool.params cannot override model width or dataset statistics.")
        try:
            json.dumps(self.params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("experiment.pool.params must contain finite JSON values.") from exc
        if "ratio" in self.params:
            validate_pool_ratio_value(self.params["ratio"])
        object.__setattr__(self, "params", deepcopy(self.params))

    def to_mapping(self) -> dict:
        return {"name": self.name, "params": deepcopy(self.params)}


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
                "Invalid experiment.training.split. Require train > 0, val > 0, "
                "and train + val < 1."
            )

    @classmethod
    def from_mapping(cls, value: dict) -> SplitConfig:
        return cls(train=float(value["train"]), val=float(value["val"]))

    def to_mapping(self) -> dict:
        return {"train": self.train, "val": self.val}


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
                raise ValueError("experiment.training.seeds.values is required when mode='list'.")
            if not self.values:
                raise ValueError("experiment.training.seeds.values must be non-empty.")
        elif self.values is not None:
            raise ValueError("experiment.training.seeds.values is only valid when mode='list'.")
        if (
            self.values is not None
            and not self.allow_duplicates
            and len(set(self.values)) != len(self.values)
        ):
            raise ValueError("Duplicate seeds require experiment.training.seeds.allow_duplicates=true.")

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
    num_runs: int
    lr: float
    batch_size: int
    patience: int
    epochs: int
    split: SplitConfig
    seeds: SeedPolicy
    # Recompute activations during backward to reduce memory; no weights are saved to disk.
    activation_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.num_runs <= 0:
            raise ValueError("experiment.training.num_runs must be positive.")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("experiment.training.lr must be a positive finite number.")
        if self.batch_size <= 0:
            raise ValueError("experiment.training.batch_size must be positive.")
        if self.patience < 0:
            raise ValueError("experiment.training.patience must be non-negative.")
        if self.epochs <= 0:
            raise ValueError("experiment.training.epochs must be positive.")
        if self.seeds.mode == "list" and self.seeds.values is not None:
            if len(self.seeds.values) != self.num_runs:
                raise ValueError("experiment.training.seeds.values length must equal experiment.training.num_runs.")

    @classmethod
    def from_mapping(cls, value: dict) -> TrainingConfig:
        return cls(
            num_runs=int(value["num_runs"]),
            lr=float(value["lr"]),
            batch_size=int(value["batch_size"]),
            patience=int(value["patience"]),
            epochs=int(value["epochs"]),
            activation_checkpoint=bool(value["activation_checkpoint"]),
            split=SplitConfig.from_mapping(value["split"]),
            seeds=SeedPolicy.from_mapping(value["seeds"]),
        )

    def to_mapping(self) -> dict:
        return {
            "num_runs": self.num_runs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "epochs": self.epochs,
            "activation_checkpoint": self.activation_checkpoint,
            "split": self.split.to_mapping(),
            "seeds": self.seeds.to_mapping(),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Dataset, pooling, model, and training settings requested for an experiment."""
    dataset: str
    pool: PoolConfig
    model: ModelConfig
    training: TrainingConfig
    # The shared graph representation exposed to pooling, independent of pre-conv consumption.
    input_type: ConnectivityType = ConnectivityType.BINARY
    compression: CompressionControl = field(default_factory=CompressionControl)

    def __post_init__(self) -> None:
        profile = get_dataset_profile(self.dataset)
        if not isinstance(self.input_type, ConnectivityType):
            raise TypeError("experiment.input_type must be a ConnectivityType.")
        if self.input_type not in profile.connectivity_types:
            raise ValueError(
                f"experiment.input_type requests {self.input_type.value} connectivity, "
                f"but dataset '{self.dataset}' cannot provide it."
            )

    @classmethod
    def from_mapping(cls, value: dict) -> ExperimentConfig:
        try:
            input_type = ConnectivityType(value.get("input_type", "binary"))
        except ValueError as exc:
            raise ValueError("experiment.input_type must be 'binary' or 'scalar'.") from exc
        return cls(
            dataset=str(value["dataset"]),
            input_type=input_type,
            compression=CompressionControl(**value.get("compression", {})),
            pool=PoolConfig(
                name=str(value["pool"]["name"]),
                params=value["pool"].get("params", {"ratio": 0.5}),
            ),
            model=ModelConfig.from_mapping(value["model"]),
            training=TrainingConfig.from_mapping(value["training"]),
        )

    def to_mapping(self) -> dict:
        return {
            "dataset": self.dataset,
            "input_type": self.input_type.value,
            "compression": self.compression.to_mapping(),
            "pool": self.pool.to_mapping(),
            "model": self.model.to_mapping(),
            "training": self.training.to_mapping(),
        }
