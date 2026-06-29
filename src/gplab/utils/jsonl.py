import json
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    # Accept UTF-8 BOM so logs produced by external tools remain readable.
    lines = [line for line in log_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Log file is empty: {path}")

    return [json.loads(line) for line in lines]


def append_jsonl(path: str, record: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file_to_save:
        file_to_save.write(json.dumps(record))
        file_to_save.write("\n")
