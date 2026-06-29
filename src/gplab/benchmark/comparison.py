from __future__ import annotations

import hashlib
import json
from typing import Optional

from .case import BenchmarkCase


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def compute_case_id(case: BenchmarkCase) -> str:
    return _hash_payload(case.to_mapping())


def benchmark_payload(case: BenchmarkCase, *, resolved_seeds: Optional[list[int]] = None) -> dict:
    training = case.training.to_mapping()
    if resolved_seeds is not None:
        training = {
            key: value
            for key, value in training.items()
            if key != "seeds"
        }
        training["seeds"] = [int(seed) for seed in resolved_seeds]

    return {
        "dataset": case.dataset,
        "model": case.model.to_mapping(),
        "pool_protocol": {
            "ratio": case.pool.ratio,
            "nonlinearity": case.pool.nonlinearity,
        },
        "training": training,
    }


def compute_benchmark_key(case: BenchmarkCase, *, resolved_seeds: Optional[list[int]] = None) -> str:
    return _hash_payload(benchmark_payload(case, resolved_seeds=resolved_seeds))


def compute_record_benchmark_key(record: dict) -> str:
    case = BenchmarkCase.from_mapping(record["case"])
    seeds = [int(seed) for seed in record["run_plan"]["seeds"]]
    return compute_benchmark_key(case, resolved_seeds=seeds)
