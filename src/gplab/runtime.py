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


def print_expr_info(
        spec: ExperimentSpec,
        device: torch.device,
        file=sys.stderr
):
    # print the information of experiments.
    if device.type == "cuda" and torch.cuda.is_available():
        device_property = torch.cuda.get_device_properties(device)
    else:
        device_property = f"CPU({platform.processor() or 'unknown'})"

    info_str = f"{sep_c('=')}\nExperiments setting:\n{spec.train.to_mapping(include_seed_path=True)}\n{sep_c('-')}\n" \
        + f"Device properties:\n{device_property}\n{sep_c('-')}\n" \
        + f"Pooling setting:\n{spec.pool.to_mapping()}\n{sep_c('-')}\n" \
        + f"Dataset:\n[green]{spec.dataset}[/green]\n{sep_c('-')}\n" \
        + f"Model configuration:\n{spec.model.to_mapping()}\n{sep_c('=')}"

    rprint(info_str, file=file)


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


def sep_c(
        sep: chr,
        ratio: float = 0.8
) -> str:
    # generate separator which fits the console width

    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 120
    w = int(ratio * columns)
    return w * sep
