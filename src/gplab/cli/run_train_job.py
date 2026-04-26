import typer
from typing_extensions import Annotated

from gplab.experiment.train_result import execute_train_job
from gplab.jobs import load_normalized_job_file
from gplab.cli.output import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    job_file: Annotated[str, typer.Option(..., help="Path to a complete automation job JSON file.")],
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "json",
):
    output_format = validate_output_format(output_format)
    try:
        normalized_job = load_normalized_job_file(job_file)
        payload = execute_train_job(
            normalized_job,
            emit_text=output_format == "text",
            request_details={
                "job_file": job_file,
                "mode": "strict_job",
                "normalized_job": normalized_job,
            },
        )

        if output_format == "json":
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("train_error", exc, details={"job_file": job_file, "mode": "strict_job"}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
