import time

import typer
from typing_extensions import Annotated, Optional

from gplab.cli.options import parse_csv_list, parse_seed_list
from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.experiment.train_result import execute_train_job
from gplab.jobs import build_case_manifest

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    pools: Annotated[str, typer.Option(help="Comma-separated pools to validate.")] = "sagpool,diffpool",
    datasets: Annotated[str, typer.Option(help="Comma-separated datasets to validate.")] = "MUTAG,PROTEINS",
    model_type: Annotated[str, typer.Option(help="Model variant: sum or plain.")] = "sum",
    pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for all cases.")] = 0.5,
    pool_nonlinearity: Annotated[str, typer.Option(help="Pooling score nonlinearity.")] = "tanh",
    activation_checkpoint: Annotated[bool, typer.Option(help="Use activation checkpointing to reduce GPU memory at extra compute cost.")] = False,
    runs: Annotated[int, typer.Option(help="Runs per smoke case.")] = 1,
    epochs: Annotated[int, typer.Option(help="Epochs per smoke case.")] = 1,
    patience: Annotated[int, typer.Option(help="Patience per smoke case.")] = 0,
    lr: Annotated[float, typer.Option(help="Learning rate per smoke case.")] = 0.0005,
    batch_size: Annotated[int, typer.Option(help="Batch size per smoke case.")] = 16,
    train_ratio: Annotated[float, typer.Option(help="Train split ratio.")] = 0.8,
    val_ratio: Annotated[float, typer.Option(help="Validation split ratio.")] = 0.1,
    log_file: Annotated[Optional[str], typer.Option(help="Optional JSONL file to append smoke records.")] = None,
    seed_mode: Annotated[Optional[str], typer.Option(help="Seed source mode override.")] = "auto",
    seed_base: Annotated[Optional[int], typer.Option(help="Seed base override.")] = 20260320,
    seed_list: Annotated[Optional[str], typer.Option(help="Comma-separated seed list override.")] = None,
    allow_duplicate_seeds: Annotated[Optional[bool], typer.Option(help="Allow duplicate seeds.")] = False,
    tag_prefix: Annotated[str, typer.Option(help="Tag prefix for smoke runs.")] = "smoke",
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        parsed_seed_list = parse_seed_list(seed_list)
        train_overrides = {
            "runs": runs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
            "epochs": epochs,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed_mode": seed_mode,
            "seed_base": seed_base,
            "allow_duplicate_seeds": allow_duplicate_seeds,
        }
        if parsed_seed_list is not None:
            train_overrides["seed_list"] = parsed_seed_list
            train_overrides["seed_mode"] = "list"

        planned_cases = build_case_manifest(
            pools=parse_csv_list(pools),
            datasets=parse_csv_list(datasets),
            model_types=[model_type],
            pool_ratio=pool_ratio,
            pool_nonlinearity=pool_nonlinearity,
            activation_checkpoint=activation_checkpoint,
            tag_prefix=tag_prefix,
            log_file=log_file,
            train_overrides=train_overrides,
        )
        cases = []
        for planned_case in planned_cases:
            pool = planned_case["pool"]
            dataset = planned_case["dataset"]
            case_id = planned_case["case_id"]
            start = time.perf_counter()
            try:
                run_payload = execute_train_job(
                    planned_case["job"],
                    emit_text=False,
                    request_details={
                        "mode": "validation",
                        "job_case_id": case_id,
                        "normalized_job": planned_case["job"],
                    },
                )
                case = {
                    "case_id": case_id,
                    "pool": pool,
                    "pool_nonlinearity": pool_nonlinearity,
                    "dataset": dataset,
                    "model_type": model_type,
                    "activation_checkpoint": activation_checkpoint,
                    "status": "ok",
                    "seconds": round(time.perf_counter() - start, 4),
                    "record_id": run_payload["summary"]["record_id"],
                    "execution": {
                        "mode": "in_process_strict_job",
                    },
                }
            except Exception as exc:
                case = {
                    "case_id": case_id,
                    "pool": pool,
                    "pool_nonlinearity": pool_nonlinearity,
                    "dataset": dataset,
                    "model_type": model_type,
                    "activation_checkpoint": activation_checkpoint,
                    "status": "failed",
                    "seconds": round(time.perf_counter() - start, 4),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "execution": {
                        "mode": "in_process_strict_job",
                    },
                }
            cases.append(case)

        summary = {
            "total": len(cases),
            "passed": sum(1 for case in cases if case["status"] == "ok"),
            "failed": sum(1 for case in cases if case["status"] == "failed"),
        }
        payload = {
            "ok": summary["failed"] == 0,
            "kind": "validation_result",
            "mode": "smoke",
            "plan": planned_cases,
            "cases": cases,
            "summary": summary,
        }

        if output_format == "json":
            emit_json(payload)
            if summary["failed"] != 0:
                raise typer.Exit(code=1)
            return

        for case in cases:
            if case["status"] == "ok":
                print(f"[{case['pool']}][{case['dataset']}] ok ({case['seconds']}s) record_id={case['record_id']}")
            else:
                print(f"[{case['pool']}][{case['dataset']}] failed ({case['seconds']}s) {case['message']}")
        print(summary)
        if summary["failed"] != 0:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("validation_error", exc))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
