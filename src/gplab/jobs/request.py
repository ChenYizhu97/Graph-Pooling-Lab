from __future__ import annotations

from gplab.benchmark.request import BenchmarkRequest

from .schema import normalize_job_shape


def request_from_job(job: dict) -> BenchmarkRequest:
    normalized = normalize_job_shape(job)
    return BenchmarkRequest.from_mapping(normalized)
