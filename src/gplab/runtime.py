import os
import sys
import platform
from datetime import datetime, timezone
import torch
from rich import print as rprint

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.execution import ExecutionOptions

try:
    import torch_geometric
except Exception:  # pragma: no cover - defensive
    torch_geometric = None


def print_experiment_info(
        case: BenchmarkCase,
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
            f"Benchmark case:\n{case.to_mapping()}",
            console_separator("-"),
            f"Execution options:\n{execution.to_mapping()}",
            console_separator("-"),
            f"Device properties:\n{device_property}",
            console_separator("="),
        ]
    )

    rprint(message, file=file)


def build_runtime_meta(device: torch.device) -> dict:
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": getattr(torch_geometric, "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    return meta


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
