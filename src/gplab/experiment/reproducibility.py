import random
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


_RUNTIME_THREADS_CONFIGURED = False


def configure_runtime_threads() -> None:
    global _RUNTIME_THREADS_CONFIGURED
    if _RUNTIME_THREADS_CONFIGURED:
        return

    # Trade-off choice for this project:
    # 1) keep CPU-side scheduling deterministic enough for repeated runs,
    # 2) avoid large training-time penalty on small TU datasets.
    # For TU datasets, setting threads/workers low usually has limited throughput impact.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads may be locked once thread pools are initialized.
        pass
    _RUNTIME_THREADS_CONFIGURED = True

    # If stricter determinism is required later, consider enabling:
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    # and setting env var CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python.


def set_np_and_torch(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return data_loader


def resolve_seeds(
    runs: int,
    seed_mode: str = "auto",
    seed_base: int = 20260320,
    seed_values: Optional[list[int]] = None,
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

    # Deterministic unique seed generation for reproducible multi-run stats.
    # We use a reproducible PRNG stream seeded by seed_base and probe until unique.
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
