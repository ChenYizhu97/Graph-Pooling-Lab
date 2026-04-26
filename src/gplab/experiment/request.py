from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional

from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_type_value,
    validate_pool_ratio_value,
    validate_pool_value,
)


@dataclass(frozen=True)
class TrainRequestContext:
    conf: dict
    log_file: Optional[str]
    seed_mode: str
    seed_base: Optional[int]
    allow_duplicate_seeds: bool
    seed_list: Optional[list[int]]


def build_cli_request(
    *,
    model_conf: dict,
    experiment_conf: dict,
    pool: Optional[str],
    pool_ratio: Optional[float],
    activation_checkpoint: Optional[bool],
    dataset_name: Optional[str],
    model_type: Optional[str],
    tag: Optional[str],
    log_file: Optional[str],
    seed_mode: str,
    seed_base: Optional[int],
    seed_list: Optional[list[int]],
    allow_duplicate_seeds: bool,
) -> TrainRequestContext:
    merged_model_conf = deepcopy(model_conf)
    merged_experiment_conf = deepcopy(experiment_conf)

    final_pool = pool if pool is not None else "nopool"
    final_pool_ratio = float(pool_ratio if pool_ratio is not None else 0.5)
    final_dataset = dataset_name if dataset_name is not None else "PROTEINS"
    final_model_type = model_type if model_type is not None else "sum"

    validate_dataset_value(final_dataset)
    validate_pool_ratio_value(final_pool_ratio)
    validate_model_type_value(final_model_type)
    is_custom_pool = validate_pool_value(final_pool)

    if "model" not in merged_model_conf:
        raise ValueError("Missing [model] section in model config")
    if "experiment" not in merged_experiment_conf:
        raise ValueError("Missing [experiment] section in experiment config")

    conf = {
        "model": deepcopy(merged_model_conf["model"]),
        "experiment": deepcopy(merged_experiment_conf["experiment"]),
    }
    expr_conf = conf["experiment"]

    runs = int(expr_conf.get("runs", 0))
    if runs <= 0:
        raise ValueError("Invalid runs value. Require experiment.runs > 0.")

    train_ratio = float(expr_conf.get("train_ratio", 0.8))
    val_ratio = float(expr_conf.get("val_ratio", 0.1))
    if (
        not math.isfinite(train_ratio)
        or not math.isfinite(val_ratio)
        or train_ratio <= 0
        or val_ratio <= 0
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError(
            "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )

    expr_conf["runs"] = runs
    expr_conf["train_ratio"] = train_ratio
    expr_conf["val_ratio"] = val_ratio
    expr_conf["seed_mode"] = seed_mode
    expr_conf["seed_base"] = seed_base
    expr_conf["seed_list"] = deepcopy(seed_list)
    expr_conf["allow_duplicate_seeds"] = allow_duplicate_seeds
    expr_conf["activation_checkpoint"] = bool(
        activation_checkpoint if activation_checkpoint is not None else expr_conf.get("activation_checkpoint", False)
    )
    conf["model"]["variant"] = final_model_type
    conf["pool"] = {
        "method": final_pool,
        "ratio": final_pool_ratio,
        "source": "custom_factory" if is_custom_pool else "builtin",
    }
    conf["dataset"] = final_dataset
    if tag is not None:
        conf["tag"] = tag

    return TrainRequestContext(
        conf=conf,
        log_file=log_file,
        seed_mode=seed_mode,
        seed_base=seed_base,
        allow_duplicate_seeds=allow_duplicate_seeds,
        seed_list=seed_list,
    )


def build_job_request(job: dict) -> TrainRequestContext:
    normalized = deepcopy(job)
    train = normalized["train"]
    pool_name = normalized["pool"]["name"]
    conf = {
        "model": deepcopy(normalized["model"]),
        "experiment": {
            "runs": train["runs"],
            "lr": train["lr"],
            "batch_size": train["batch_size"],
            "patience": train["patience"],
            "epochs": train["epochs"],
            "train_ratio": train["train_ratio"],
            "val_ratio": train["val_ratio"],
            "seed_mode": train["seed_mode"],
            "seed_base": train["seed_base"],
            "seed_list": deepcopy(train["seed_list"]),
            "allow_duplicate_seeds": train["allow_duplicate_seeds"],
            "activation_checkpoint": train["activation_checkpoint"],
        },
        "pool": {
            "method": pool_name,
            "ratio": normalized["pool"]["ratio"],
            "source": "custom_factory" if ":" in pool_name else "builtin",
        },
        "dataset": normalized["dataset"],
    }
    if normalized["tag"] is not None:
        conf["tag"] = normalized["tag"]

    return TrainRequestContext(
        conf=conf,
        log_file=normalized["log_file"],
        seed_mode=train["seed_mode"],
        seed_base=train["seed_base"],
        allow_duplicate_seeds=train["allow_duplicate_seeds"],
        seed_list=deepcopy(train["seed_list"]),
    )
