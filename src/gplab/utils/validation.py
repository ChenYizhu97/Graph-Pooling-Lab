import math

from gplab.utils.registry import BUILTIN_POOLS, TU_DATASETS

MODEL_TYPES = ("sum", "plain")
SEED_MODES = ("auto", "file", "list")


def validate_dataset_value(name: str) -> None:
    if name not in TU_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {', '.join(TU_DATASETS)}")


def validate_pool_value(name: str, builtins: tuple[str, ...] = BUILTIN_POOLS) -> bool:
    is_custom_pool = ":" in name
    if not is_custom_pool and name not in builtins:
        raise ValueError(f"Unknown pooling method '{name}'. Built-ins: {', '.join(builtins)}")
    return is_custom_pool


def validate_pool_ratio_value(ratio: float) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("pool_ratio must be in (0, 1].")
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("pool_ratio must be in (0, 1].")


def validate_model_type_value(value: str) -> None:
    if value not in MODEL_TYPES:
        raise ValueError("model_type must be 'sum' or 'plain'.")


def validate_seed_mode_value(mode: str, *, allowed: tuple[str, ...] = SEED_MODES) -> None:
    if mode not in allowed:
        raise ValueError(f"seed_mode must be one of: {', '.join(allowed)}.")


def normalize_config_seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("experiment.seed_list in config must be a list of integers.")
    return int(value)
