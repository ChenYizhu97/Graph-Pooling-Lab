from typing import Any, Optional

from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.execute import run_experiment
from gplab.experiment.record import summarize_record
from gplab.utils.jsonl import append_jsonl


def execute_train_request(
    request: BenchmarkRequest,
    *,
    emit_text: bool,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record = run_experiment(request, emit_text=emit_text)
    if request.execution.log_file is not None:
        append_jsonl(request.execution.log_file, record)

    context_payload: dict[str, Any] = {}
    if context:
        context_payload.update(context)

    payload = {
        "ok": True,
        "kind": "train_result",
        "record": record,
        "summary": summarize_record(record),
        "context": context_payload,
    }
    if emit_text:
        summary = payload["summary"]
        print(
            f"Result: mean={summary['mean']:.4f} std={summary['std']:.4f} "
            f"record_id={summary['record_id']}"
        )
    return payload
