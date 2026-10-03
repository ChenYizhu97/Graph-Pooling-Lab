import math
from copy import deepcopy
from typing import Optional

from gplab.utils.validation import validate_seed_mode_value

from .defaults import (
    AUTOMATION_MODEL_DEFAULTS,
    AUTOMATION_TRAINING_DEFAULTS,
)

JOB_TOP_LEVEL_FIELDS = {"experiment", "log_file", "tag", "runs", "source_record_id"}
JOB_REQUIRED_TOP_LEVEL_FIELDS = {"experiment"}
EXPERIMENT_FIELDS = {"dataset", "pool", "model", "training"}
EXPERIMENT_REQUIRED_FIELDS = {"dataset", "pool", "training"}
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


class JobSchemaError(ValueError):
    """JSON boundary error carrying field-level details for automation clients."""
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


def _normalize_fields(payload: dict, label: str, validators: dict) -> dict:
    """Validate flat fields in declaration order, retaining fully qualified error paths."""
    return {name: validate(payload[name], field_name=f"{label}.{name}")
            for name, validate in validators.items()}


def normalize_job_shape(job: dict) -> dict:
    """Fill defaults and enforce JSON types before domain configuration is built.

    Config constructors coerce values, so this boundary must first reject e.g.
    booleans as numbers and strings as integers. Domain validation then checks
    ranges and cross-field constraints. Deep copies isolate mutable defaults;
    nested split/seed defaults are merged separately to support partial objects.
    """
    raw = require_mapping(job, label="job")
    _reject_unknown_fields(raw, allowed=JOB_TOP_LEVEL_FIELDS, label="top-level")
    _require_keys(raw, required=JOB_REQUIRED_TOP_LEVEL_FIELDS, label="top-level")

    experiment = require_mapping(raw["experiment"], label="experiment")
    _reject_unknown_fields(experiment, allowed=EXPERIMENT_FIELDS, label="experiment")
    _require_keys(experiment, required=EXPERIMENT_REQUIRED_FIELDS, label="experiment")

    pool = {
        **deepcopy(POOL_DEFAULTS),
        **require_mapping(experiment["pool"], label="experiment.pool"),
    }
    _reject_unknown_fields(pool, allowed=POOL_FIELDS, label="experiment.pool")
    _require_keys(pool, required=POOL_REQUIRED_FIELDS, label="experiment.pool")

    raw_model = require_mapping(experiment.get("model", {}), label="experiment.model")
    model = {
        **deepcopy(AUTOMATION_MODEL_DEFAULTS),
        **raw_model,
    }
    _reject_unknown_fields(model, allowed=MODEL_FIELDS, label="experiment.model")

    raw_training = require_mapping(experiment.get("training", {}), label="experiment.training")
    _require_keys(raw_training, required=TRAINING_REQUIRED_FIELDS, label="experiment.training")
    training = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS),
        **raw_training,
    }
    _reject_unknown_fields(training, allowed=TRAINING_FIELDS, label="experiment.training")

    split = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS["split"]),
        **require_mapping(training["split"], label="experiment.training.split"),
    }
    _reject_unknown_fields(split, allowed=SPLIT_FIELDS, label="experiment.training.split")

    seeds = {
        **deepcopy(AUTOMATION_TRAINING_DEFAULTS["seeds"]),
        **require_mapping(training["seeds"], label="experiment.training.seeds"),
    }
    _reject_unknown_fields(seeds, allowed=SEED_FIELDS, label="experiment.training.seeds")

    normalized = {
        "experiment": {
            "dataset": _require_string(experiment["dataset"], field_name="experiment.dataset"),
            "pool": _normalize_fields(pool, "experiment.pool", {
                "name": _require_string, "ratio": _normalize_float,
                "nonlinearity": _require_string,
            }),
            "model": _normalize_fields(model, "experiment.model", {
                "hidden_features": _normalize_int, "nonlinearity": _require_string,
                "p_dropout": _normalize_float, "pre_conv": _require_string,
                "post_conv": _require_string, "pre_gnn": _normalize_int_list,
                "post_gnn": _normalize_int_list, "variant": _require_string,
            }),
            "training": {
                **_normalize_fields(training, "experiment.training", {
                    "runs": _normalize_int, "lr": _normalize_float,
                    "batch_size": _normalize_int, "patience": _normalize_int,
                    "epochs": _normalize_int, "activation_checkpoint": _normalize_bool,
                }),
                "split": _normalize_fields(split, "experiment.training.split", {
                    "train": _normalize_float, "val": _normalize_float,
                }),
                "seeds": {
                    "mode": _require_string(seeds["mode"], field_name="experiment.training.seeds.mode"),
                    "base": _normalize_int(seeds["base"], field_name="experiment.training.seeds.base"),
                    "values": None if seeds["values"] is None else _normalize_int_list(
                        seeds["values"], field_name="experiment.training.seeds.values", allow_empty=False,
                    ),
                    "allow_duplicates": _normalize_bool(
                        seeds["allow_duplicates"], field_name="experiment.training.seeds.allow_duplicates",
                    ),
                },
            },
        },
    }

    normalized.update({
        field: _normalize_optional_string(raw.get(field), field_name=field)
        for field in ("log_file", "tag", "source_record_id")
    })
    normalized["runs"] = _normalize_runs(raw.get("runs"))

    try:
        validate_seed_mode_value(normalized["experiment"]["training"]["seeds"]["mode"])
    except ValueError as exc:
        raise JobSchemaError(
            str(exc),
            field="experiment.training.seeds.mode",
            expected="one of: auto, list",
        ) from exc
    return normalized


def _normalize_runs(value) -> list[dict] | None:
    """Validate explicit replay runs; dataset-dependent partition checks happen at preparation."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise JobSchemaError("runs must be a non-empty array.", field="runs", expected="non-empty array")
    runs = []
    for index, value in enumerate(value):
        label = f"runs[{index}]"
        run = require_mapping(value, label=label)
        _reject_unknown_fields(run, allowed={"seed", "split"}, label=label)
        _require_keys(run, required={"seed", "split"}, label=label)
        split = require_mapping(run["split"], label=f"{label}.split")
        _reject_unknown_fields(split, allowed={"train", "val", "test"}, label=f"{label}.split")
        _require_keys(split, required={"train", "val", "test"}, label=f"{label}.split")
        runs.append({
            "seed": _normalize_int(run["seed"], field_name=f"{label}.seed"),
            "split": {name: _normalize_int_list(split[name], field_name=f"{label}.split.{name}",
                                                allow_empty=False)
                      for name in ("train", "val", "test")},
        })
    return runs
