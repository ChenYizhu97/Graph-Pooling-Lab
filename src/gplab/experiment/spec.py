from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

from gplab.utils.registry import SUPPORTED_CONVS
from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_type_value,
    validate_pool_ratio_value,
    validate_pool_value,
    validate_seed_mode_value,
)


@dataclass(frozen=True)
class ModelSpec:
    hidden_features: int
    nonlinearity: str
    p_dropout: float
    conv_layer: str
    pre_gnn: tuple[int, ...]
    post_gnn: tuple[int, ...]
    variant: str

    def __post_init__(self) -> None:
        if self.hidden_features <= 0:
            raise ValueError("model.hidden_features must be positive.")
        if not self.nonlinearity:
            raise ValueError("model.nonlinearity must be non-empty.")
        if not 0.0 <= self.p_dropout < 1.0:
            raise ValueError("model.p_dropout must be in [0, 1).")
        if self.conv_layer not in SUPPORTED_CONVS:
            raise ValueError(
                f"Unsupported model.conv_layer '{self.conv_layer}'. "
                f"Supported layers: {', '.join(SUPPORTED_CONVS)}."
            )
        if not self.pre_gnn or any(width <= 0 for width in self.pre_gnn):
            raise ValueError("model.pre_gnn must be a non-empty array of positive integers.")
        if self.pre_gnn[-1] != self.hidden_features:
            raise ValueError(
                "model.pre_gnn must end with model.hidden_features so conv1 receives the configured width."
            )
        if not self.post_gnn or any(width <= 0 for width in self.post_gnn):
            raise ValueError("model.post_gnn must be a non-empty array of positive integers.")
        expected_readout_width = 2 * self.hidden_features
        if self.post_gnn[0] != expected_readout_width:
            raise ValueError(
                f"model.post_gnn must start with {expected_readout_width}, "
                "the concatenated add/max readout width."
            )
        validate_model_type_value(self.variant)

    @classmethod
    def from_mapping(cls, value: dict) -> ModelSpec:
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
class PoolSpec:
    name: str
    ratio: float
    nonlinearity: str = "tanh"

    def __post_init__(self) -> None:
        validate_pool_value(self.name)
        validate_pool_ratio_value(self.ratio)
        if not self.nonlinearity:
            raise ValueError("pool.nonlinearity must be non-empty.")

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
class TrainSpec:
    runs: int
    lr: float
    batch_size: int
    patience: int
    epochs: int
    train_ratio: float
    val_ratio: float
    seed_mode: str
    seed_base: int
    seed_list: Optional[tuple[int, ...]]
    allow_duplicate_seeds: bool
    activation_checkpoint: bool
    seeds_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("train.runs must be positive.")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("train.lr must be a positive finite number.")
        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be positive.")
        if self.patience < 0:
            raise ValueError("train.patience must be non-negative.")
        if self.epochs <= 0:
            raise ValueError("train.epochs must be positive.")
        if (
            not math.isfinite(self.train_ratio)
            or not math.isfinite(self.val_ratio)
            or self.train_ratio <= 0
            or self.val_ratio <= 0
            or self.train_ratio + self.val_ratio >= 1
        ):
            raise ValueError(
                "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, "
                "and train_ratio + val_ratio < 1."
            )
        validate_seed_mode_value(self.seed_mode)
        if self.seed_mode == "list":
            if self.seed_list is None:
                raise ValueError("train.seed_list is required when train.seed_mode='list'.")
            if len(self.seed_list) != self.runs:
                raise ValueError("train.seed_list length must equal train.runs.")
        elif self.seed_list is not None:
            raise ValueError("train.seed_list is only valid when train.seed_mode='list'.")
        if self.seed_mode == "file" and not self.seeds_path:
            raise ValueError("A seed file path is required when seed_mode='file'.")
        if (
            self.seed_list is not None
            and not self.allow_duplicate_seeds
            and len(set(self.seed_list)) != len(self.seed_list)
        ):
            raise ValueError("Duplicate seeds require allow_duplicate_seeds=true.")

    @classmethod
    def from_mapping(cls, value: dict, *, seeds_path: Optional[str] = None) -> TrainSpec:
        raw_seed_list = value.get("seed_list")
        return cls(
            runs=int(value["runs"]),
            lr=float(value["lr"]),
            batch_size=int(value["batch_size"]),
            patience=int(value["patience"]),
            epochs=int(value["epochs"]),
            train_ratio=float(value["train_ratio"]),
            val_ratio=float(value["val_ratio"]),
            seed_mode=str(value["seed_mode"]),
            seed_base=int(value["seed_base"]),
            seed_list=None if raw_seed_list is None else tuple(int(seed) for seed in raw_seed_list),
            allow_duplicate_seeds=bool(value["allow_duplicate_seeds"]),
            activation_checkpoint=bool(value["activation_checkpoint"]),
            seeds_path=seeds_path,
        )

    def to_mapping(self, *, include_seed_path: bool = False) -> dict:
        value = {
            "runs": self.runs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "epochs": self.epochs,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "seed_mode": self.seed_mode,
            "seed_base": self.seed_base,
            "seed_list": None if self.seed_list is None else list(self.seed_list),
            "allow_duplicate_seeds": self.allow_duplicate_seeds,
            "activation_checkpoint": self.activation_checkpoint,
        }
        if include_seed_path:
            value["seeds_path"] = self.seeds_path
        return value


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    pool: PoolSpec
    model: ModelSpec
    train: TrainSpec
    log_file: Optional[str] = None
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        validate_dataset_value(self.dataset)

    @classmethod
    def from_job(cls, job: dict) -> ExperimentSpec:
        train = TrainSpec.from_mapping(job["train"])
        if train.seed_mode == "file":
            raise ValueError("Strict automation jobs support seed_mode 'auto' or 'list', not 'file'.")
        return cls(
            dataset=job["dataset"],
            pool=PoolSpec(
                name=job["pool"]["name"],
                ratio=float(job["pool"]["ratio"]),
                nonlinearity=job["pool"].get("nonlinearity", "tanh"),
            ),
            model=ModelSpec.from_mapping(job["model"]),
            train=train,
            log_file=job["log_file"],
            tag=job["tag"],
        )

    @classmethod
    def from_record(cls, record: dict, *, log_file: Optional[str] = None) -> ExperimentSpec:
        spec = record["spec"]
        train = spec["train"]
        seeds = tuple(int(seed) for seed in train["seeds"])
        return cls(
            dataset=spec["dataset"],
            pool=PoolSpec(
                name=spec["pool"]["name"],
                ratio=float(spec["pool"]["ratio"]),
                nonlinearity=spec["pool"].get("nonlinearity", "tanh"),
            ),
            model=ModelSpec.from_mapping(spec["model"]),
            train=TrainSpec(
                runs=len(seeds),
                lr=float(train["lr"]),
                batch_size=int(train["batch_size"]),
                patience=int(train["patience"]),
                epochs=int(train["epochs"]),
                train_ratio=float(train["split"]["train"]),
                val_ratio=float(train["split"]["val"]),
                seed_mode="list",
                seed_base=20260320,
                seed_list=seeds,
                allow_duplicate_seeds=len(set(seeds)) != len(seeds),
                activation_checkpoint=bool(train.get("activation_checkpoint", False)),
            ),
            log_file=log_file,
            tag=record.get("tag"),
        )

    def to_job(self) -> dict:
        if self.train.seed_mode == "file":
            raise ValueError("Strict automation jobs cannot serialize seed_mode='file'.")
        return {
            "dataset": self.dataset,
            "pool": self.pool.to_mapping(),
            "model": self.model.to_mapping(),
            "train": self.train.to_mapping(),
            "log_file": self.log_file,
            "tag": self.tag,
        }

    def to_record_spec(self, resolved_seeds: list[int]) -> dict:
        split = {
            "train": self.train.train_ratio,
            "val": self.train.val_ratio,
            "test": 1.0 - self.train.train_ratio - self.train.val_ratio,
        }
        return {
            "dataset": self.dataset,
            "model": self.model.to_mapping(),
            "pool": {
                **self.pool.to_mapping(),
                "source": self.pool.source,
            },
            "train": {
                "lr": self.train.lr,
                "batch_size": self.train.batch_size,
                "patience": self.train.patience,
                "epochs": self.train.epochs,
                "activation_checkpoint": self.train.activation_checkpoint,
                "split": split,
                "seeds": [int(seed) for seed in resolved_seeds],
            },
        }

    def request_metadata(self) -> dict:
        return {
            "log_file": self.log_file,
            "seed_mode": self.train.seed_mode,
            "seed_base": self.train.seed_base,
            "allow_duplicate_seeds": self.train.allow_duplicate_seeds,
            "seed_list": None if self.train.seed_list is None else list(self.train.seed_list),
            "activation_checkpoint": self.train.activation_checkpoint,
        }
