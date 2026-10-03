"""Execute a submitted job and return the record optionally written to JSONL."""
from gplab.experiment.execute import run_experiment
from gplab.experiment.record import build_record, summarize_record
from gplab.jobs.job import ExperimentJob
from gplab.utils.jsonl import append_jsonl


def execute_job(
    job: ExperimentJob,
    *,
    emit_text: bool,
    context: dict | None = None,
) -> dict:
    """Train, finalize the record, optionally save it, and build the CLI response."""
    measurements = run_experiment(
        job.experiment, job.execution, fixed_runs=job.fixed_runs, emit_text=emit_text,
    )
    record = build_record(
        job.experiment, execution=job.execution, **measurements,
        tag=job.tag, source_record_id=job.source_record_id,
    )
    if job.log_file is not None:
        append_jsonl(job.log_file, record)
    summary = summarize_record(record)
    if emit_text:
        print(
            f"Result: mean={summary['mean']:.4f} std={summary['std']:.4f} "
            f"record_id={summary['record_id']}"
        )
    return {
        "ok": True,
        "kind": "train_result",
        "record": record,
        "summary": summary,
        "context": dict(context or {}),
    }
