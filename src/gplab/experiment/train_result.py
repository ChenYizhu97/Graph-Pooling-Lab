from typing import Any, Optional

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.execution import ExecutionOptions
from gplab.experiment.execute import run_experiment
from gplab.experiment.record import summarize_record
from gplab.experiment.builders import build_job_request
from gplab.utils.jsonl import append_jsonl


def execute_train_request(
    case: BenchmarkCase,
    execution: ExecutionOptions,
    *,
    emit_text: bool,
    request_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record = run_experiment(case, execution, emit_text=emit_text)
    if execution.log_file is not None:
        append_jsonl(execution.log_file, record)

    request_payload: dict[str, Any] = {}
    if request_details:
        request_payload.update(request_details)
    request_payload.update(execution.request_metadata())

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
    case, execution = build_job_request(job)
    return execute_train_request(
        case,
        execution,
        emit_text=emit_text,
        request_details=request_details,
    )
