"""Describe the active software and device environment; this module does not configure it."""
import os
import platform
import sys
from datetime import datetime, timezone

import torch
import torch_geometric
from rich import print as rprint

from gplab.benchmark.config import ExperimentConfig
from gplab.benchmark.execution import ExecutionOptions


def print_experiment_info(
        experiment: ExperimentConfig,
        execution: ExecutionOptions,
        device: torch.device,
        file=sys.stderr
):
    if device.type == "cuda" and torch.cuda.is_available():
        device_property = torch.cuda.get_device_properties(device)
    else:
        device_property = f"CPU({platform.processor() or 'unknown'})"

    message = "\n".join(
        [
            console_separator("="),
            f"Benchmark experiment:\n{experiment.to_mapping()}",
            console_separator("-"),
            f"Execution options:\n{execution.to_mapping()}",
            console_separator("-"),
            f"Device properties:\n{device_property}",
            console_separator("="),
        ]
    )

    rprint(message, file=file)


def collect_environment_info(device: torch.device) -> dict:
    """Snapshot versions and effective backend settings after execution setup."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": getattr(torch_geometric, "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def console_separator(
        character: str,
        width_ratio: float = 0.8
) -> str:
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 120
    width = int(width_ratio * columns)
    return width * character
