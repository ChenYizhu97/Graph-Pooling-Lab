from copy import deepcopy
import math
from typing import Optional

from gplab.utils.validation import validate_seed_mode_value

from .defaults import AUTOMATION_EXECUTION_DEFAULTS, AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAINING_DEFAULTS


JOB_TOP_LEVEL_FIELDS = {"case", "execution"}
JOB_REQUIRED_TOP_LEVEL_FIELDS = {"case"}
CASE_FIELDS = {"dataset", "pool", "model", "training"}
CASE_REQUIRED_FIELDS = {"dataset", "pool", "training"}
POOL_FIELDS = {"name", "ratio", "nonlinearity"}
POOL_REQUIRED_FIELDS = {"name", "ratio"}
POOL_DEFAULTS = {
    "nonlinearity": "tanh",
}
MODEL_FIELDS = set(AUTOMATION_MODEL_DEFAULTS)
TRAINING_FIELDS = set(AUTOMATION_TRAINING_DEFAULTS)
TRAINING_REQUIRED_FIELDS = {"runs", "epochs", "patience"}
SPLIT_FIELDS = {"train", "val"}
SEED_FIELDS = {"mode", "base", "values", "allow_duplicates"}
EXECUTION_FIELDS = set(AUTOMATION_EXECUTION_DEFAULTS)


class JobSchemaError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        expected: str | None = None,
        missing: list[str] | None = None,
        unknown: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected
        self.missing = missing
        self.unknown = unknown


def require_mapping(value, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise JobSchemaError(
            f"{label} must be a JSON object.",
            field=label,
            expected="JSON object",
        )
    return value


def _reject_unknown_fields(payload: dict, *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise JobSchemaError(
            f"Unknown {label} field(s): {joined}.",
            field=label,
            expected=f"allowed fields: {', '.join(sorted(allowed))}",
            unknown=unknown,
        )


def _require_keys(payload: dict, *, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise JobSchemaError(
            f"Missing required {label} field(s): {joined}.",
            field=label,
            expected=f"required fields: {', '.join(sorted(required))}",
            missing=missing,
        )


def _normalize_optional_string(value, *, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise JobSchemaError(
            f"{field_name} must be a string or null.",
            field=field_name,
            expected="string or null",
        )
    return value


def _require_string(value, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise JobSchemaError(
            f"{field_name} must be a string.",
            field=field_name,
            expected="string",
        )
    return value


def _normalize_bool(value, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise JobSchemaError(
            f"{field_name} must be a boolean.",
            field=field_name,
            expected="boolean",
        )
    return value


def _normalize_int(value, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JobSchemaError(
            f"{field_name} must be an integer.",
            field=field_name,
            expected="integer",
        )
    return int(value)


def _normalize_int_list(value, *, field_name: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list):
        raise JobSchemaError(
            f"{field_name} must be an array of integers.",
            field=field_name,
            expected="array of integers",
        )
    if not allow_empty and not value:
        raise JobSchemaError(
            f"{field_name} must be a non-empty array of integers.",
            field=field_name,
            expected="non-empty array of integers",
        )
    return [_normalize_int(item, field_name=f"{field_name}[]") for item in value]


def _normalize_float(value, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobSchemaError(
            f"{field_name} must be a number.",
            field=field_name,
            expected="finite number",
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise JobSchemaError(
            f"{field_name} must be a finite number.",
            field=field_name,
            expected="finite number",
        )
    return normalized


def normalize_job_shape(job: dict) -> dict:
    raw = require_mapping(job, label="job")
    _reject_unknown_fields(raw, allowed=JOB_TOP_LEVEL_FIELDS, label="top-level")
    _require_keys(raw, required=JOB_REQUIRED_TOP_LEVEL_FIELDS, label="top-level")

    case = require_mapping(raw["case"], label="case")
    _reject_unknown_fields(case, allowed=CASE_FIELDS, label="case")
    _require_keys(case, required=CASE_REQUIRED_FIELDS, label="case")

    pool = {
        **deepcopy(POOL_DEFAULTS),
        **require_mapping(case["pool"], label="case.pool"),
    }
    _reject_unknown_fields(pool, allowed=POOL_FIELDS, label="case.pool")
    _require_keys(pool, required=POOL_REQUIRED_FIELDS, label="case.pool")

    model = {
        **deepcopy(AUTOMATION_MODEL_DEFAULTS),
        **require_mapping(case.get("model", {}), label="case.model"),
    }
    _reject_unknown_fields(model, allowed=MODEL_FIELDS, label="case.model")

    raw_training = require_mapping(case.get("training", {}), label="case.training")
    _require_keys(raw_training, required=TRAINING_REQUIRED_FIELDS, label="case.training")
    training = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS),
        **raw_training,
    }
    _reject_unknown_fields(training, allowed=TRAINING_FIELDS, label="case.training")

    split = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS["split"]),
        **require_mapping(training["split"], label="case.training.split"),
    }
    _reject_unknown_fields(split, allowed=SPLIT_FIELDS, label="case.training.split")

    seeds = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS["seeds"]),
        **require_mapping(training["seeds"], label="case.training.seeds"),
    }
    _reject_unknown_fields(seeds, allowed=SEED_FIELDS, label="case.training.seeds")

    execution = {
        **deepcopy(AUTOMATION_EXECUTION_DEFAULTS),
        **require_mapping(raw.get("execution", {}), label="execution"),
    }
    _reject_unknown_fields(execution, allowed=EXECUTION_FIELDS, label="execution")

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

    try:
        validate_seed_mode_value(normalized["case"]["training"]["seeds"]["mode"])
    except ValueError as exc:
        raise JobSchemaError(
            str(exc),
            field="case.training.seeds.mode",
            expected="one of: auto, list",
        ) from exc
    return normalized
