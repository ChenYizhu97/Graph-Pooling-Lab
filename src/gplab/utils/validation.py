import math

MODEL_VARIANTS = ("sum", "plain")
SEED_MODES = ("auto", "list")


def validate_pool_ratio_value(ratio: float) -> None:
    # TGP/PyG use values below one as fractions and values >= one as node counts.
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio) or ratio <= 0 or (ratio >= 1 and int(ratio) != ratio)):
        raise ValueError("experiment.pool.params.ratio must be a fraction in (0, 1) or a positive integer count.")


def validate_model_variant_value(value: str) -> None:
    if value not in MODEL_VARIANTS:
        raise ValueError("model_variant must be 'sum' or 'plain'.")


def validate_seed_mode_value(mode: str, *, allowed: tuple[str, ...] = SEED_MODES) -> None:
    if mode not in allowed:
        raise ValueError(f"seed_mode must be one of: {', '.join(allowed)}.")


def normalize_config_seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("training.seeds.values in config must be a list of integers.")
    return int(value)
