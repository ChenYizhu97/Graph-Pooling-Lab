"""Record the core software versions and device used by an experiment."""
import sys
from importlib.metadata import version

import torch
import torch_geometric


def collect_environment_info(device: torch.device) -> dict:
    """Capture actual installed backend versions, including the pooling implementation."""
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "tgp_version": version("torch-geometric-pool"),
        "device": str(device),
    }
