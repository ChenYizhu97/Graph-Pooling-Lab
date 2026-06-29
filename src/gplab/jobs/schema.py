import math
from typing import Optional

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.comparison import compute_case_id
from gplab.benchmark.execution import ExecutionOptions
from gplab.utils.validation import validate_seed_mode_value

from .defaults import AUTOMATION_EXECUTION_DEFAULTS, AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAINING_DEFAULTS


JOB_TOP_LEVEL_FIELDS = {"case", "execution"}
CASE_FIELDS = {"dataset", "pool", "model", "training"}
POOL_FIELDS = {"name", "ratio", "nonlinearity"}
MODEL_FIELDS = set(AUTOMATION_MODEL_DEFAULTS)
TRAINING_FIELDS = set(AUTOMATION_TRAINING_DEFAULTS)
SPLIT_FIELDS = {"train", "val"}
SEED_FIELDS = {"mode", "base", "values", "allow_duplicates"}
EXECUTION_FIELDS = set(AUTOMATION_EXECUTION_DEFAULTS)


def require_mapping(value, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _reject_unknown_fields(payload: dict, *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {label} field(s): {joined}.")


def _require_keys(payload: dict, *, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required {label} field(s): {joined}.")


def _normalize_optional_string(value, *, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    return value


def _require_string(value, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value


def _normalize_bool(value, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _normalize_int(value, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _normalize_int_list(value, *, field_name: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array of integers.")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be a non-empty array of integers.")
    return [_normalize_int(item, field_name=f"{field_name}[]") for item in value]


def _normalize_float(value, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number.")
    return normalized


def normalize_train_job(job: dict) -> dict:
    raw = require_mapping(job, label="job")
    _reject_unknown_fields(raw, allowed=JOB_TOP_LEVEL_FIELDS, label="top-level")
    _require_keys(raw, required=JOB_TOP_LEVEL_FIELDS, label="top-level")

    case = require_mapping(raw["case"], label="case")
    _reject_unknown_fields(case, allowed=CASE_FIELDS, label="case")
    _require_keys(case, required=CASE_FIELDS, label="case")

    pool = require_mapping(case["pool"], label="case.pool")
    _reject_unknown_fields(pool, allowed=POOL_FIELDS, label="case.pool")
    _require_keys(pool, required=POOL_FIELDS, label="case.pool")

    model = require_mapping(case["model"], label="case.model")
    _reject_unknown_fields(model, allowed=MODEL_FIELDS, label="case.model")
    _require_keys(model, required=MODEL_FIELDS, label="case.model")

    training = require_mapping(case["training"], label="case.training")
    _reject_unknown_fields(training, allowed=TRAINING_FIELDS, label="case.training")
    _require_keys(training, required=TRAINING_FIELDS, label="case.training")

    split = require_mapping(training["split"], label="case.training.split")
    _reject_unknown_fields(split, allowed=SPLIT_FIELDS, label="case.training.split")
    _require_keys(split, required=SPLIT_FIELDS, label="case.training.split")

    seeds = require_mapping(training["seeds"], label="case.training.seeds")
    _reject_unknown_fields(seeds, allowed=SEED_FIELDS, label="case.training.seeds")
    _require_keys(seeds, required=SEED_FIELDS, label="case.training.seeds")

    execution = require_mapping(raw["execution"], label="execution")
    _reject_unknown_fields(execution, allowed=EXECUTION_FIELDS, label="execution")
    _require_keys(execution, required=EXECUTION_FIELDS, label="execution")

    normalized = {
        "case": {
            "dataset": _require_string(case["dataset"], field_name="case.dataset"),
            "pool": {
                "name": _require_string(pool["name"], field_name="case.pool.name"),
                "ratio": _normalize_float(pool["ratio"], field_name="case.pool.ratio"),
                "nonlinearity": _require_string(
                    pool["nonlinearity"],
                    field_name="case.pool.nonlinearity",
                ),
            },
            "model": {
                "hidden_features": _normalize_int(
                    model["hidden_features"],
                    field_name="case.model.hidden_features",
                ),
                "nonlinearity": _require_string(
                    model["nonlinearity"],
                    field_name="case.model.nonlinearity",
                ),
                "p_dropout": _normalize_float(model["p_dropout"], field_name="case.model.p_dropout"),
                "conv_layer": _require_string(model["conv_layer"], field_name="case.model.conv_layer"),
                "pre_gnn": _normalize_int_list(model["pre_gnn"], field_name="case.model.pre_gnn"),
                "post_gnn": _normalize_int_list(model["post_gnn"], field_name="case.model.post_gnn"),
                "variant": _require_string(model["variant"], field_name="case.model.variant"),
            },
            "training": {
                "runs": _normalize_int(training["runs"], field_name="case.training.runs"),
                "lr": _normalize_float(training["lr"], field_name="case.training.lr"),
                "batch_size": _normalize_int(
                    training["batch_size"],
                    field_name="case.training.batch_size",
                ),
                "patience": _normalize_int(training["patience"], field_name="case.training.patience"),
                "epochs": _normalize_int(training["epochs"], field_name="case.training.epochs"),
                "split": {
                    "train": _normalize_float(split["train"], field_name="case.training.split.train"),
                    "val": _normalize_float(split["val"], field_name="case.training.split.val"),
                },
                "seeds": {
                    "mode": _require_string(seeds["mode"], field_name="case.training.seeds.mode"),
                    "base": _normalize_int(seeds["base"], field_name="case.training.seeds.base"),
                    "values": None
                    if seeds["values"] is None
                    else _normalize_int_list(
                        seeds["values"],
                        field_name="case.training.seeds.values",
                        allow_empty=False,
                    ),
                    "allow_duplicates": _normalize_bool(
                        seeds["allow_duplicates"],
                        field_name="case.training.seeds.allow_duplicates",
                    ),
                },
            },
        },
        "execution": {
            "log_file": _normalize_optional_string(execution["log_file"], field_name="execution.log_file"),
            "tag": _normalize_optional_string(execution["tag"], field_name="execution.tag"),
            "activation_checkpoint": _normalize_bool(
                execution["activation_checkpoint"],
                field_name="execution.activation_checkpoint",
            ),
        },
    }

    validate_seed_mode_value(normalized["case"]["training"]["seeds"]["mode"])
    case_obj = BenchmarkCase.from_mapping(normalized["case"])
    execution_obj = ExecutionOptions.from_mapping(normalized["execution"])
    return {
        "case": case_obj.to_mapping(),
        "execution": execution_obj.to_mapping(),
    }


def compute_train_job_case_id(job: dict) -> str:
    normalized = normalize_train_job(job)
    return compute_case_id(BenchmarkCase.from_mapping(normalized["case"]))
