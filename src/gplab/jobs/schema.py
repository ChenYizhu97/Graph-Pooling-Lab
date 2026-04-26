import hashlib
import json
import math
from typing import Optional

from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_type_value,
    validate_pool_ratio_value,
    validate_pool_value,
)

from .defaults import AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAIN_DEFAULTS


JOB_TOP_LEVEL_FIELDS = {"dataset", "pool", "model", "train", "log_file", "tag"}
JOB_POOL_FIELDS = {"name", "ratio"}
FULL_MODEL_FIELDS = set(AUTOMATION_MODEL_DEFAULTS)
FULL_TRAIN_FIELDS = set(AUTOMATION_TRAIN_DEFAULTS)


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

    pool = require_mapping(raw["pool"], label="pool")
    _reject_unknown_fields(pool, allowed=JOB_POOL_FIELDS, label="pool")
    _require_keys(pool, required=JOB_POOL_FIELDS, label="pool")

    model = require_mapping(raw["model"], label="model")
    _reject_unknown_fields(model, allowed=FULL_MODEL_FIELDS, label="model")
    _require_keys(model, required=FULL_MODEL_FIELDS, label="model")

    train = require_mapping(raw["train"], label="train")
    _reject_unknown_fields(train, allowed=FULL_TRAIN_FIELDS, label="train")
    _require_keys(train, required=FULL_TRAIN_FIELDS, label="train")

    normalized = {
        "dataset": _require_string(raw["dataset"], field_name="dataset"),
        "pool": {
            "name": _require_string(pool["name"], field_name="pool.name"),
            "ratio": _normalize_float(pool["ratio"], field_name="pool.ratio"),
        },
        "model": {
            "hidden_features": _normalize_int(model["hidden_features"], field_name="model.hidden_features"),
            "nonlinearity": _require_string(model["nonlinearity"], field_name="model.nonlinearity"),
            "p_dropout": _normalize_float(model["p_dropout"], field_name="model.p_dropout"),
            "conv_layer": _require_string(model["conv_layer"], field_name="model.conv_layer"),
            "pre_gnn": _normalize_int_list(model["pre_gnn"], field_name="model.pre_gnn"),
            "post_gnn": _normalize_int_list(model["post_gnn"], field_name="model.post_gnn"),
            "variant": _require_string(model["variant"], field_name="model.variant"),
        },
        "train": {
            "runs": _normalize_int(train["runs"], field_name="train.runs"),
            "lr": _normalize_float(train["lr"], field_name="train.lr"),
            "batch_size": _normalize_int(train["batch_size"], field_name="train.batch_size"),
            "patience": _normalize_int(train["patience"], field_name="train.patience"),
            "epochs": _normalize_int(train["epochs"], field_name="train.epochs"),
            "train_ratio": _normalize_float(train["train_ratio"], field_name="train.train_ratio"),
            "val_ratio": _normalize_float(train["val_ratio"], field_name="train.val_ratio"),
            "seed_mode": _require_string(train["seed_mode"], field_name="train.seed_mode"),
            "seed_base": _normalize_int(train["seed_base"], field_name="train.seed_base"),
            "seed_list": None
            if train["seed_list"] is None
            else _normalize_int_list(train["seed_list"], field_name="train.seed_list", allow_empty=False),
            "allow_duplicate_seeds": _normalize_bool(
                train["allow_duplicate_seeds"],
                field_name="train.allow_duplicate_seeds",
            ),
            "activation_checkpoint": _normalize_bool(
                train["activation_checkpoint"],
                field_name="train.activation_checkpoint",
            ),
        },
        "log_file": _normalize_optional_string(raw["log_file"], field_name="log_file"),
        "tag": _normalize_optional_string(raw["tag"], field_name="tag"),
    }

    validate_dataset_value(normalized["dataset"])
    validate_pool_ratio_value(normalized["pool"]["ratio"])
    validate_pool_value(normalized["pool"]["name"])
    validate_model_type_value(normalized["model"]["variant"])

    if normalized["train"]["seed_mode"] not in {"auto", "file", "list"}:
        raise ValueError("train.seed_mode must be 'auto', 'file', or 'list'.")

    train_ratio = normalized["train"]["train_ratio"]
    val_ratio = normalized["train"]["val_ratio"]
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    if normalized["train"]["runs"] <= 0:
        raise ValueError("Invalid runs value. Require train.runs > 0.")
    if normalized["train"]["seed_list"] is not None and normalized["train"]["seed_mode"] != "list":
        raise ValueError("train.seed_mode must be 'list' when train.seed_list is provided in a complete job.")

    return normalized


def compute_train_job_case_id(job: dict) -> str:
    payload = {
        "dataset": job["dataset"],
        "pool": {
            "name": job["pool"]["name"],
            "ratio": job["pool"]["ratio"],
        },
        "model": {
            "hidden_features": job["model"]["hidden_features"],
            "nonlinearity": job["model"]["nonlinearity"],
            "p_dropout": job["model"]["p_dropout"],
            "conv_layer": job["model"]["conv_layer"],
            "pre_gnn": job["model"]["pre_gnn"],
            "post_gnn": job["model"]["post_gnn"],
            "variant": job["model"]["variant"],
        },
        "train": {
            "runs": job["train"]["runs"],
            "lr": job["train"]["lr"],
            "batch_size": job["train"]["batch_size"],
            "patience": job["train"]["patience"],
            "epochs": job["train"]["epochs"],
            "train_ratio": job["train"]["train_ratio"],
            "val_ratio": job["train"]["val_ratio"],
            "seed_mode": job["train"]["seed_mode"],
            "seed_base": job["train"]["seed_base"],
            "seed_list": job["train"]["seed_list"],
            "allow_duplicate_seeds": job["train"]["allow_duplicate_seeds"],
            "activation_checkpoint": job["train"]["activation_checkpoint"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]
