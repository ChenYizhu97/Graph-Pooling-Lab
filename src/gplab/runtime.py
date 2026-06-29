import os
import sys
import platform
from datetime import datetime, timezone
import torch
from rich import print as rprint

from gplab.experiment.spec import ExperimentSpec

try:
    import torch_geometric
except Exception:  # pragma: no cover - defensive
    torch_geometric = None


def print_experiment_info(
        spec: ExperimentSpec,
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
            f"Experiments setting:\n{spec.train.to_mapping(include_seed_path=True)}",
            console_separator("-"),
            f"Device properties:\n{device_property}",
            console_separator("-"),
            f"Pooling setting:\n{spec.pool.to_mapping()}",
            console_separator("-"),
            f"Dataset:\n[green]{spec.dataset}[/green]",
            console_separator("-"),
            f"Model configuration:\n{spec.model.to_mapping()}",
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
