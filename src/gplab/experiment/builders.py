from typing import Optional

from gplab.benchmark.case import (
    BenchmarkCase,
    ModelConfig,
    PoolConfig,
    SeedPolicy,
    SplitConfig,
    TrainingConfig,
)
from gplab.benchmark.execution import ExecutionOptions


def build_cli_request(
    *,
    model_config: dict,
    training_config: dict,
    execution_config: dict,
    pool: Optional[str],
    pool_ratio: Optional[float],
    pool_nonlinearity: Optional[str],
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
) -> tuple[BenchmarkCase, ExecutionOptions]:
    if "model" not in model_config:
        raise ValueError("Missing [model] section in model config.")
    if "training" not in training_config:
        raise ValueError("Missing [training] section in experiment config.")

    model_section = dict(model_config["model"])
    model_section["variant"] = model_variant or model_section.get("variant", "sum")

    training_section = dict(training_config["training"])
    split_section = dict(training_section.get("split", {}))

    case = BenchmarkCase(
        dataset=dataset_name or "PROTEINS",
        pool=PoolConfig(
            name=pool or "nopool",
            ratio=float(pool_ratio if pool_ratio is not None else 0.5),
            nonlinearity=pool_nonlinearity or "tanh",
        ),
        model=ModelConfig.from_mapping(model_section),
        training=TrainingConfig(
            runs=int(training_section["runs"]),
            lr=float(training_section["lr"]),
            batch_size=int(training_section["batch_size"]),
            patience=int(training_section["patience"]),
            epochs=int(training_section["epochs"]),
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

    execution_defaults = dict(execution_config.get("execution", {}))
    execution = ExecutionOptions(
        log_file=log_file if log_file is not None else execution_defaults.get("log_file"),
        tag=tag if tag is not None else execution_defaults.get("tag"),
        activation_checkpoint=bool(
            activation_checkpoint
            if activation_checkpoint is not None
            else execution_defaults.get("activation_checkpoint", False)
        ),
    )
    return case, execution


def build_job_request(job: dict) -> tuple[BenchmarkCase, ExecutionOptions]:
    return BenchmarkCase.from_mapping(job["case"]), ExecutionOptions.from_mapping(job["execution"])
