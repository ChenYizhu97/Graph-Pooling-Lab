from typing import Annotated, Optional

import torch
import typer

from gplab.benchmark.request import BenchmarkRequest
from gplab.cli.output import (
    build_error_payload,
    emit_json,
    redirect_stdout_for_json,
    validate_output_format,
)
from gplab.experiment.record import summarize_record
from gplab.experiment.record_log import RecordLogError, find_record_by_id, load_record_log
from gplab.experiment.train_result import execute_train_request
from gplab.runtime import build_runtime_meta

app = typer.Typer(pretty_exceptions_enable=False)


def _compatibility_status(recorded: dict, current: dict) -> tuple[str, list[dict]]:
    checks = [
        ("python_version", "python"),
        ("torch_version", "torch"),
        ("torch_geometric_version", "torch_geometric"),
        ("device", "device"),
        ("cuda_available", "cuda_available"),
    ]
    details = [
        {"field": label, "recorded": recorded[key], "current": current[key],
         "match": recorded[key] == current[key]}
        for key, label in checks
    ]
    status = "compatible" if all(item["match"] for item in details) else "mismatch"
    return status, details


@app.command()
def main(
    log_file: Annotated[str, typer.Option(..., help="JSONL log file containing the record to replay.")],
    record_id: Annotated[str, typer.Option(help="Record id of the JSONL entry to replay.")] = ...,
    replay_log_file: Annotated[
        Optional[str],
        typer.Option(help="Optional JSONL file to append the replayed result to."),
    ] = None,
    run: Annotated[bool, typer.Option(help="Execute the replay in this process.")] = False,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    json_output = output_format == "json"
    try:
        record = find_record_by_id(load_record_log(log_file), record_id)
        replay_request = BenchmarkRequest.from_record_for_replay(record, replay_log_file=replay_log_file)
        replay_job = replay_request.to_mapping()
        # Keep source and replay IDs distinct: replacing auto seeds with an
        # explicit list changes case identity even when the actual runs match.
        source_case_id = record["run_plan"]["case_id"]
        replay_case_id = replay_request.case_id

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        current_runtime = build_runtime_meta(device)
        status, details = _compatibility_status(record["runtime"], current_runtime)
        replay_payload = {
            "ok": True,
            "kind": "replay_result",
            "record": summarize_record(record),
            "job": replay_job,
            "context": {
                "source": "record_replay",
                "case_id": replay_case_id,
                "source_case_id": source_case_id,
            },
            "paths": {
                "replay_log_file": replay_log_file,
            },
            "compatibility": {
                "status": status,
                "details": details,
            },
        }

        if not json_output:
            print(f"Replay record: {record['record_id']}")
            print("Replay mode: in-process record replay")
            print(f"Source case_id: {source_case_id}")
            print(f"Replay job case_id: {replay_case_id}")
            if replay_log_file is not None:
                print(f"Replay log file: {replay_log_file}")
            if status == "compatible":
                print("Runtime compatibility: current environment matches recorded runtime on checked fields.")
            else:
                print(f"Runtime compatibility: {status}")
                for item in details:
                    if not item["match"]:
                        print(f"  - {item['field']}: recorded={item['recorded']!r}, current={item['current']!r}")

        if run:
            with redirect_stdout_for_json(json_output):
                run_payload = execute_train_request(
                    replay_request,
                    emit_text=not json_output,
                    context={
                        "source": "record_replay",
                        "source_record_id": record["record_id"],
                        "source_case_id": source_case_id,
                        "case_id": replay_case_id,
                        "job": replay_job,
                    },
                )
            replay_payload["rerun"] = {
                "requested": True,
                "ok": True,
                "payload": run_payload,
                "record_id": run_payload["summary"]["record_id"],
                "summary": run_payload["summary"],
                "appended_to_log": replay_log_file is not None,
            }
            if not json_output:
                print(f"Rerun record_id: {run_payload['summary']['record_id']}")
        elif not json_output:
            print("Use --run to execute this replay.")
        if json_output:
            emit_json(replay_payload)
    except typer.Exit:
        raise
    except RecordLogError as exc:
        if json_output:
            emit_json(build_error_payload("replay_error", exc, details={"log_file": log_file, "record_id": record_id}))
            raise typer.Exit(code=1)
        param_hint = f"--{exc.field.replace('_', '-')}" if exc.field == "record_id" else None
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc
    except Exception as exc:
        if json_output:
            emit_json(build_error_payload("replay_error", exc, details={"log_file": log_file, "record_id": record_id}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
