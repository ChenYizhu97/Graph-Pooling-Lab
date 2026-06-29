from typing import Optional

import typer

from gplab.utils.validation import (
    normalize_config_seed,
    validate_seed_mode_value,
)


def _raise_bad_parameter(exc: ValueError, *, param_hint: str) -> None:
    raise typer.BadParameter(str(exc), param_hint=param_hint) from exc


def parse_seed_list(seed_list: Optional[str]) -> Optional[list[int]]:
    if seed_list is None:
        return None

    raw_tokens = [token.strip() for token in seed_list.split(",")]
    values = [token for token in raw_tokens if token]
    if not values:
        raise typer.BadParameter("seed_list must contain at least one integer.", param_hint="--seed-list")

    try:
        return [int(token) for token in values]
    except ValueError as exc:
        raise typer.BadParameter(
            "seed_list must be a comma-separated list of integers.",
            param_hint="--seed-list",
        ) from exc


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    values = [item for item in items if item]
    if not values:
        raise typer.BadParameter("Expected at least one non-empty comma-separated value.")
    return values


def resolve_seed_options(
    seed_mode: Optional[str],
    seed_base: Optional[int],
    seed_list: Optional[str],
    allow_duplicate_seeds: Optional[bool],
    experiment_section: dict,
) -> tuple[str, int, bool, Optional[list[int]]]:
    experiment_defaults = experiment_section or {}
    final_seed_list = parse_seed_list(seed_list)

    if final_seed_list is None:
        configured_seed_list = experiment_defaults.get("seed_list")
        if configured_seed_list is not None:
            if not isinstance(configured_seed_list, list):
                raise typer.BadParameter(
                    "experiment.seed_list in config must be a list of integers.",
                    param_hint="--seed-list",
                )
            try:
                final_seed_list = [normalize_config_seed(value) for value in configured_seed_list]
            except ValueError as exc:
                _raise_bad_parameter(exc, param_hint="--seed-list")

    final_seed_mode = (
        seed_mode
        if seed_mode is not None
        else experiment_defaults.get("seed_mode", "auto")
    )
    if final_seed_list is not None:
        final_seed_mode = "list"
    try:
        validate_seed_mode_value(final_seed_mode)
    except ValueError as exc:
        _raise_bad_parameter(exc, param_hint="--seed-mode")

    final_seed_base = (
        seed_base
        if seed_base is not None
        else experiment_defaults.get("seed_base", 20260320)
    )
    if final_seed_base is None:
        final_seed_base = 20260320

    allow_duplicates = allow_duplicate_seeds
    if allow_duplicates is None:
        allow_duplicates = experiment_defaults.get("allow_duplicate_seeds", False)
    allow_duplicates = bool(allow_duplicates)

    return final_seed_mode, int(final_seed_base), allow_duplicates, final_seed_list
