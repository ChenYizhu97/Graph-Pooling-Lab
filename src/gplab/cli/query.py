import typer
from typing_extensions import Annotated, Optional

from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.experiment.query import (
    QuerySpec,
    QuerySpecError,
    build_benchmark_report,
    build_query_result,
    format_query_text,
    format_report_text,
)
from gplab.experiment.record_log import load_record_log

app = typer.Typer(pretty_exceptions_enable=False)


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
        typer.Option(help="Sort field: mean, std, avg_best_epoch, avg_val_loss."),
    ] = "mean",
    show_case: Annotated[bool, typer.Option(help="Include the full case block in default output.")] = False,
    show_replay: Annotated[bool, typer.Option(help="Show gplab-replay command for each matched record.")] = False,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        spec = QuerySpec(
            log_file=log_file,
            dataset=dataset,
            pool=pool,
            model_variant=model_variant,
            tag=tag,
            sort_by=sort_by,
            show_case=show_case,
            show_replay=show_replay,
        )
        records = load_record_log(log_file)

        payload = build_benchmark_report(records, spec) if report else build_query_result(records, spec)
        if output_format == "json":
            emit_json(payload)
            return

        text = format_report_text(payload) if report else format_query_text(payload)
        if text:
            print(text)
    except typer.Exit:
        raise
    except QuerySpecError as exc:
        if output_format == "json":
            emit_json(build_error_payload("query_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        param_hint = f"--{exc.field.replace('_', '-')}" if exc.field is not None else None
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("query_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
