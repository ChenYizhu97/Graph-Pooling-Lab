import typer
from typing_extensions import Annotated

from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.jobs import compute_train_job_case_id, load_normalized_job_file

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    job_file: Annotated[str, typer.Option(..., help="Path to a job JSON file.")],
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "json",
):
    output_format = validate_output_format(output_format)
    try:
        normalized_job = load_normalized_job_file(job_file)
        payload = {
            "ok": True,
            "kind": "normalized_job",
            "case_id": compute_train_job_case_id(normalized_job),
            "job": normalized_job,
        }
        if output_format == "json":
            emit_json(payload)
            return
        print(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("normalize_job_error", exc, details={"job_file": job_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
