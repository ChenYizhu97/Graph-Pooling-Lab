import sys

import typer
from typing_extensions import Annotated, Optional

from gplab.experiment.train_result import execute_train_request
from gplab.jobs import load_job_file, load_job_text, request_from_job
from gplab.cli.output import build_error_payload, emit_json, validate_output_format

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
    try:
        job = _load_job_input(job_file=job_file, job_json=job_json, job_stdin=job_stdin)
        request = request_from_job(job)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            details = {"source": "job_json"}
            if job_file is not None:
                details["job_file"] = job_file
            emit_json(build_error_payload("job_error", exc, details=details))
            raise typer.Exit(code=1)
        raise

    try:
        context = {
            "source": "job_json",
            "case_id": request.case_id,
        }
        if job_file is not None:
            context["job_file"] = job_file
        payload = execute_train_request(
            request,
            emit_text=output_format == "text",
            context=context,
        )

        if output_format == "json":
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            details = {
                "source": "job_json",
                "case_id": request.case_id,
            }
            if job_file is not None:
                details["job_file"] = job_file
            emit_json(
                build_error_payload(
                    "train_error",
                    exc,
                    details=details,
                )
            )
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
