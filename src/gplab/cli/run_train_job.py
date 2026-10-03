import sys
from typing import Annotated, Optional

import typer

from gplab.cli.output import (
    build_error_payload,
    emit_json,
    redirect_stdout_for_json,
    validate_output_format,
)
from gplab.experiment.train_result import execute_train_request
from gplab.jobs import load_job_file, load_job_text, request_from_job

app = typer.Typer(pretty_exceptions_enable=False)


def _load_job_input(
    *,
    job_file: str | None,
    job_json: str | None,
    job_stdin: bool,
) -> dict:
    selected_count = sum(value is not None for value in (job_file, job_json)) + int(job_stdin)
    if selected_count != 1:
        raise typer.BadParameter(
            "Provide exactly one of --job-file, --job-json, or --job-stdin.",
            param_hint="--job-file/--job-json/--job-stdin",
        )
    if job_file is not None:
        return load_job_file(job_file)
    if job_json is not None:
        return load_job_text(job_json, label="job JSON from --job-json")
    return load_job_text(sys.stdin.read(), label="job JSON from stdin")


@app.command()
def main(
    job_file: Annotated[
        Optional[str],
        typer.Option(help="Path to an automation Job JSON file."),
    ] = None,
    job_json: Annotated[
        Optional[str],
        typer.Option(help="Inline automation Job JSON."),
    ] = None,
    job_stdin: Annotated[
        bool,
        typer.Option(help="Read automation Job JSON from stdin."),
    ] = False,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "json",
):
    output_format = validate_output_format(output_format)
    json_output = output_format == "json"
    context = {"source": "job_json"}
    if job_file is not None:
        context["job_file"] = job_file
    error_kind = "job_error"
    try:
        with redirect_stdout_for_json(json_output):
            job = _load_job_input(job_file=job_file, job_json=job_json, job_stdin=job_stdin)
            request = request_from_job(job)
            context["case_id"] = request.case_id
            # Preserve the response contract: parsing/validation failures are
            # job_error; failures after a request is accepted are train_error.
            error_kind = "train_error"
            payload = execute_train_request(request, emit_text=not json_output, context=context)
        if json_output:
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if json_output:
            emit_json(build_error_payload(error_kind, exc, details=context))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
