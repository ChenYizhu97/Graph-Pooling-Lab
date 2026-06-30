import json
from pathlib import Path

from .schema import require_mapping


def load_job_file(path: str) -> dict:
    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {path}")

    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid job JSON: {exc}") from exc

    return require_mapping(payload, label="job")
