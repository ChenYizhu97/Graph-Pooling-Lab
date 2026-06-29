import hashlib
import json

from gplab.benchmark.comparison import compute_record_benchmark_key


def compute_record_id(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "record_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def attach_record_id(record: dict) -> dict:
    record["record_id"] = compute_record_id(record)
    return record


def require_record_id(record: dict) -> dict:
    if "record_id" not in record:
        raise ValueError("Record is missing required field: record_id.")
    return record


def compute_benchmark_key(record: dict) -> str:
    return compute_record_benchmark_key(record)
