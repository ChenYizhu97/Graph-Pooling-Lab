from __future__ import annotations

from gplab.benchmark.request import BenchmarkRequest

from .schema import JobSchemaError, normalize_job_shape


def _infer_error_field(message: str) -> str | None:
    if message.startswith("case."):
        return message.split()[0]
    if "dataset" in message:
        return "case.dataset"
    if "pooling method" in message:
        return "case.pool.name"
    if "pool_ratio" in message:
        return "case.pool.ratio"
    if "model_variant" in message:
        return "case.model.variant"
    if "seed_mode" in message:
        return "case.training.seeds.mode"
    return None


def request_from_job(job: dict) -> BenchmarkRequest:
    normalized = normalize_job_shape(job)
    try:
        return BenchmarkRequest.from_mapping(normalized)
    except ValueError as exc:
        raise JobSchemaError(str(exc), field=_infer_error_field(str(exc))) from exc
