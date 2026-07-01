import json
from pathlib import Path

from .schema import require_mapping


def load_job_text(text: str, *, label: str = "job JSON") -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc

    return require_mapping(payload, label="job")


def load_job_file(path: str) -> dict:
    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {path}")

    return load_job_text(job_path.read_text(encoding="utf-8"), label=f"job JSON in {path}")
