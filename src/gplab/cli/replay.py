from typing import Optional

import torch
import typer
from typing_extensions import Annotated

from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.identity import require_record_id
from gplab.experiment.record import summarize_record
from gplab.experiment.train_result import execute_train_request
from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.runtime import build_runtime_meta
from gplab.utils.jsonl import read_jsonl

app = typer.Typer(pretty_exceptions_enable=False)


def _compatibility_status(recorded: dict, current: dict) -> tuple[str, list[dict]]:
    checks = [
        ("python_version", "python"),
        ("torch_version", "torch"),
        ("torch_geometric_version", "torch_geometric"),
        ("device", "device"),
        ("cuda_available", "cuda_available"),
    ]
    details = []
    mismatch_found = False
    for key, label in checks:
        recorded_value = recorded[key]
        current_value = current[key]
        match = recorded_value == current_value
        if recorded_value != current_value:
            mismatch_found = True
        details.append(
            {
                "field": label,
                "recorded": recorded_value,
                "current": current_value,
                "match": match,
            }
        )

    if mismatch_found:
        return "mismatch", details
    return "compatible", details


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
    try:
        records = [require_record_id(record) for record in read_jsonl(log_file)]
        record = next((record for record in records if record["record_id"] == record_id), None)
        if record is None:
            raise typer.BadParameter(
                f"record_id '{record_id}' was not found in the selected log file.",
                param_hint="--record-id",
            )
        replay_request = BenchmarkRequest.from_record_for_replay(record, replay_log_file=replay_log_file)
        replay_job = replay_request.to_mapping()
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
            "execution": {
                "mode": "in_process_strict_job",
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

        if output_format == "json":
            if run:
                run_payload = execute_train_request(
                    replay_request,
                    emit_text=False,
                    request_details={
                        "mode": "replay",
                        "source_record_id": record["record_id"],
                        "source_case_id": source_case_id,
                        "job_case_id": replay_case_id,
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
            emit_json(replay_payload)
            return

        print(f"Replay record: {record['record_id']}")
        print("Replay mode: in-process strict job")
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
            run_payload = execute_train_request(
                replay_request,
                emit_text=True,
                request_details={
                    "mode": "replay",
                    "source_record_id": record["record_id"],
                    "source_case_id": source_case_id,
                    "job_case_id": replay_case_id,
                    "job": replay_job,
                },
            )
            print(f"Rerun record_id: {run_payload['summary']['record_id']}")
        else:
            print("Use --run to execute this replay.")
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("replay_error", exc, details={"log_file": log_file, "record_id": record_id}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
