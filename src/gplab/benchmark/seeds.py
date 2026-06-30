from __future__ import annotations

import numpy as np


def resolve_seeds(
    runs: int,
    seed_mode: str = "auto",
    seed_base: int = 20260320,
    seed_values: list[int] | None = None,
    allow_duplicate_seeds: bool = False,
) -> list[int]:
    if runs <= 0:
        return []

    if seed_mode not in {"auto", "list"}:
        raise ValueError("seed_mode must be 'auto' or 'list'.")

    if seed_mode == "list":
        if seed_values is None:
            raise ValueError("seed_values is required when seed_mode='list'.")
        if len(seed_values) != runs:
            raise ValueError(
                f"seed_values length must equal runs. Got {len(seed_values)} seeds for runs={runs}."
            )
        if (not allow_duplicate_seeds) and (len(set(seed_values)) != len(seed_values)):
            raise ValueError(
                "Duplicate seeds detected in list mode. "
                "Set allow_duplicate_seeds=true only when intentionally replaying duplicate seeds."
            )
        return [int(seed) for seed in seed_values]

    rng = np.random.default_rng(seed_base)
    generated: list[int] = []
    used = set()
    while len(generated) < runs:
        candidate = int(rng.integers(1, 2**31 - 1))
        if candidate in used:
            continue
        used.add(candidate)
        generated.append(candidate)
    return generated
