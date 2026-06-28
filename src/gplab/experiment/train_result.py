from typing import Any, Optional

from gplab.experiment.execute import run_experiment
from gplab.experiment.record import summarize_record
from gplab.experiment.builders import build_job_spec
from gplab.experiment.spec import ExperimentSpec
from gplab.utils.jsonl import append_jsonl


def execute_train_request(
    spec: ExperimentSpec,
    *,
    emit_text: bool,
    request_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record = run_experiment(spec, emit_text=emit_text)
    if spec.log_file is not None:
        append_jsonl(spec.log_file, record)

    request_payload: dict[str, Any] = {}
    if request_details:
        request_payload.update(request_details)
    request_payload.update(spec.request_metadata())

    payload = {
        "ok": True,
        "kind": "train_result",
        "record": record,
        "summary": summarize_record(record),
        "request": request_payload,
    }
    if emit_text:
        summary = payload["summary"]
        print(
            f"Result: mean={summary['mean']:.4f} std={summary['std']:.4f} "
            f"record_id={summary['record_id']}"
        )
    return payload


def execute_train_job(
    job: dict,
    *,
    emit_text: bool,
    request_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return execute_train_request(
        build_job_spec(job),
        emit_text=emit_text,
        request_details=request_details,
    )
