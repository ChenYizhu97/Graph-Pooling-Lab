import shlex

import typer
from typing_extensions import Annotated, Optional

from gplab.benchmark.comparison import compute_record_benchmark_key
from gplab.experiment.identity import require_record_id
from gplab.experiment.record import summarize_record
from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.utils.jsonl import read_jsonl
from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_variant_value,
    validate_pool_value,
)

app = typer.Typer(pretty_exceptions_enable=False)


def _sort_value(record: dict, sort_by: str) -> float:
    summary = summarize_record(record)
    return float(summary[sort_by])


def _rank_groups(records: list[dict], sort_by: str) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(compute_record_benchmark_key(record), []).append(record)
    return [
        (
            benchmark_key,
            sorted(
                group,
                key=lambda record: _sort_value(record, sort_by),
                reverse=sort_by not in {"std", "avg_val_loss", "avg_best_epoch"},
            ),
        )
        for benchmark_key, group in groups.items()
    ]


def _print_report(records: list[dict], sort_by: str) -> None:
    for benchmark_key, ranked in _rank_groups(records, sort_by):
        first = ranked[0]
        case = first["case"]
        tags = sorted(
            {
                record["execution"].get("tag")
                for record in ranked
                if record["execution"].get("tag") is not None
            }
        )
        header_parts = [
            f"dataset={case['dataset']}",
            f"model={case['model']['variant']}",
            f"ratio={case['pool']['ratio']}",
            f"pool_nonlinearity={case['pool']['nonlinearity']}",
            f"benchmark={benchmark_key}",
        ]
        if len(tags) == 1:
            header_parts.append(f"tag={tags[0]}")
        elif len(tags) > 1:
            header_parts.append(f"tags={len(tags)}")
        print(" | ".join(header_parts))
        for index, record in enumerate(ranked, start=1):
            summary = summarize_record(record)
            corr = summary["val_loss_test_acc_corr"]
            corr_text = "n/a" if corr is None else f"{corr:.4f}"
            print(
                f"{index}. pool={summary['pool']} ratio={summary['pool_ratio']} "
                f"mean={summary['mean']:.4f} std={summary['std']:.4f} "
                f"avg_epoch={summary['avg_best_epoch']:.1f} avg_val_loss={summary['avg_val_loss']:.6f} "
                f"val_test_corr={corr_text} record_id={summary['record_id']}"
            )
        print(
            "Interpretation: compare mean first, then use std for stability, avg_epoch for early-stop behavior, "
            "and val_test_corr to judge whether lower validation loss aligned with better test accuracy."
        )
        print()


def _build_report_payload(records: list[dict], sort_by: str) -> dict:
    payload_groups = []
    for benchmark_key, ranked in _rank_groups(records, sort_by):
        group_summaries = []
        for index, record in enumerate(ranked, start=1):
            summary = summarize_record(record)
            summary["rank"] = index
            group_summaries.append(summary)

        first = ranked[0]
        payload_groups.append(
            {
                "benchmark_key": benchmark_key,
                "dataset": first["case"]["dataset"],
                "model_variant": first["case"]["model"]["variant"],
                "pool_ratio": first["case"]["pool"]["ratio"],
                "pool_nonlinearity": first["case"]["pool"]["nonlinearity"],
                "records": group_summaries,
            }
        )

    return {
        "ok": True,
        "kind": "query_report",
        "groups": payload_groups,
    }


@app.command()
def main(
    log_file: Annotated[str, typer.Option(..., help="JSONL log file to query.")],
    pool: Annotated[Optional[str], typer.Option()] = None,
    dataset: Annotated[Optional[str], typer.Option()] = None,
    model_variant: Annotated[
        Optional[str],
        typer.Option(help="Filter by model variant: sum or plain."),
    ] = None,
    tag: Annotated[Optional[str], typer.Option(help="Filter by execution tag.")] = None,
    report: Annotated[
        bool,
        typer.Option(help="Print grouped benchmark report instead of one summary dict per record."),
    ] = False,
    sort_by: Annotated[
        str,
        typer.Option(help="Report sort field: mean, std, avg_best_epoch, avg_val_loss."),
    ] = "mean",
    show_case: Annotated[bool, typer.Option(help="Include the full case block in default output.")] = False,
    show_replay: Annotated[bool, typer.Option(help="Show gplab-replay command for each matched record.")] = False,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        if sort_by not in {"mean", "std", "avg_best_epoch", "avg_val_loss"}:
            raise typer.BadParameter(
                "sort_by must be one of: mean, std, avg_best_epoch, avg_val_loss.",
                param_hint="--sort-by",
            )

        if dataset is not None:
            try:
                validate_dataset_value(dataset)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--dataset") from exc
        if pool is not None:
            try:
                validate_pool_value(pool)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--pool") from exc
        if model_variant is not None:
            try:
                validate_model_variant_value(model_variant)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--model-variant") from exc

        records = []
        for record in (require_record_id(record) for record in read_jsonl(log_file)):
            case = record["case"]
            if dataset is not None and case["dataset"].lower() != dataset.lower():
                continue
            if pool is not None and case["pool"]["name"] != pool:
                continue
            if tag is not None and record["execution"].get("tag") != tag:
                continue
            if model_variant is not None and case["model"]["variant"] != model_variant:
                continue
            records.append(record)

        if report:
            if output_format == "json":
                emit_json(_build_report_payload(records, sort_by))
                return
            _print_report(records, sort_by)
            return

        summaries = []
        for record in records:
            summary = summarize_record(record)
            if show_case:
                summary["case"] = record["case"]
            if show_replay:
                summary["replay_command"] = (
                    f"gplab-replay --log-file {shlex.quote(log_file)} --record-id {record['record_id']}"
                )
            summaries.append(summary)

        if output_format == "json":
            emit_json({"ok": True, "kind": "query_result", "records": summaries})
            return

        for summary in summaries:
            print(summary)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("query_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
