from typing import Optional

from gplab.benchmark.compression import CompressionControl
from gplab.benchmark.config import (
    ExperimentConfig,
    ModelConfig,
    PoolConfig,
    SeedPolicy,
    SplitConfig,
    TrainingConfig,
)
from gplab.graph import ConnectivityType
from gplab.jobs.job import ExperimentJob


def build_cli_job(
    *,
    model_config: dict,
    training_config: dict,
    pool: Optional[str],
    pool_params: Optional[dict],
    activation_checkpoint: Optional[bool],
    dataset_name: Optional[str],
    model_variant: Optional[str],
    tag: Optional[str],
    log_file: Optional[str],
    seed_mode: str,
    seed_base: int,
    seed_values: Optional[list[int]],
    allow_duplicate_seeds: bool,
    split_train: Optional[float],
    split_val: Optional[float],
    input_type: Optional[str] = None,
    compression_mode: Optional[str] = None,
    target_retention: Optional[float] = None,
) -> ExperimentJob:
    """Apply CLI overrides to TOML defaults, then construct validated benchmark values."""
    if "model" not in model_config:
        raise ValueError("Missing [model] section in model config.")
    if "training" not in training_config:
        raise ValueError("Missing [training] section in experiment config.")

    model_section = dict(model_config["model"])
    model_section["variant"] = model_variant or model_section.get("variant", "sum")

    training_section = dict(training_config["training"])
    split_section = dict(training_section.get("split", {}))

    compression = dict(training_config.get("compression", {}))
    if compression_mode is not None:
        compression["mode"] = compression_mode
        if compression_mode == "native":
            compression.pop("target_retention", None)
    if target_retention is not None:
        compression["target_retention"] = target_retention

    experiment = ExperimentConfig(
        dataset=dataset_name or "PROTEINS",
        compression=CompressionControl(**compression),
        input_type=ConnectivityType(input_type if input_type is not None else training_config.get("input_type", "binary")),
        pool=PoolConfig(
            name=pool or "nopool",
            params=pool_params if pool_params is not None else {"ratio": 0.5},
        ),
        model=ModelConfig.from_mapping(model_section),
        training=TrainingConfig(
            num_runs=int(training_section["num_runs"]),
            lr=float(training_section["lr"]),
            batch_size=int(training_section["batch_size"]),
            patience=int(training_section["patience"]),
            epochs=int(training_section["epochs"]),
            activation_checkpoint=bool(
                activation_checkpoint if activation_checkpoint is not None
                else training_section.get("activation_checkpoint", False)
            ),
            split=SplitConfig(
                train=float(split_train if split_train is not None else split_section["train"]),
                val=float(split_val if split_val is not None else split_section["val"]),
            ),
            seeds=SeedPolicy(
                mode=seed_mode,
                base=seed_base,
                values=None if seed_values is None else tuple(seed_values),
                allow_duplicates=allow_duplicate_seeds,
            ),
        ),
    )

    return ExperimentJob(experiment=experiment, log_file=log_file, tag=tag)
