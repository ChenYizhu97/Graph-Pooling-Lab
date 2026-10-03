from __future__ import annotations

from gplab.jobs.job import ExperimentJob

from .schema import JobSchemaError, normalize_job_shape


def _infer_error_field(message: str) -> str | None:
    if message.startswith("experiment."):
        return message.split()[0]
    if "dataset" in message:
        return "experiment.dataset"
    if "pooling method" in message:
        return "experiment.pool.name"
    if "pool_ratio" in message:
        return "experiment.pool.ratio"
    if "model_variant" in message:
        return "experiment.model.variant"
    if "seed_mode" in message:
        return "experiment.training.seeds.mode"
    return None


def parse_job(job: dict) -> ExperimentJob:
    """Validate JSON shape and domain rules, preserving field information on errors."""
    normalized = normalize_job_shape(job)
    try:
        return ExperimentJob.from_mapping(normalized)
    except ValueError as exc:
        raise JobSchemaError(str(exc), field=_infer_error_field(str(exc))) from exc
