from typing import Any, Optional

from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.execute import run_experiment
from gplab.experiment.record import summarize_record
from gplab.utils.jsonl import append_jsonl


def persist_record(record: dict, log_file: Optional[str]) -> None:
    if log_file is not None:
        append_jsonl(log_file, record)


def build_train_result(
    record: dict,
    *,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    context_payload: dict[str, Any] = {}
    if context:
        context_payload.update(context)

    return {
        "ok": True,
        "kind": "train_result",
        "record": record,
        "summary": summarize_record(record),
        "context": context_payload,
    }


def print_train_result_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(
        f"Result: mean={summary['mean']:.4f} std={summary['std']:.4f} "
        f"record_id={summary['record_id']}"
    )


def execute_train_request(
    request: BenchmarkRequest,
    *,
    emit_text: bool,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record = run_experiment(request, emit_text=emit_text)
    persist_record(record, request.execution.log_file)
    payload = build_train_result(record, context=context)
    if emit_text:
        print_train_result_summary(payload)
    return payload
