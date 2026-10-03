import random

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

_RUNTIME_THREADS_CONFIGURED = False


def configure_runtime_threads() -> None:
    """Use one compute thread for small TU graphs; configure interop once if still possible."""
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


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN behavior."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Build a zero-worker loader with an independent generator for repeatable shuffling."""
    # Isolate loader randomness from model initialization and stochastic layers,
    # so their random draws do not change the seeded training shuffle.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
