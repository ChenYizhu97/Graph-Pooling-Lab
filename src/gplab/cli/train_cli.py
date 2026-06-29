import toml
import typer
from typing_extensions import Annotated, Optional

from gplab.cli.options import resolve_seed_options
from gplab.experiment.builders import build_cli_spec
from gplab.experiment.train_result import execute_train_request
from gplab.paths import default_config_path
from gplab.cli.output import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    pool: Annotated[
        Optional[str],
        typer.Option(help="Pooling method name or <module:factory> for custom pooling."),
    ] = None,
    pool_ratio: Annotated[
        Optional[float],
        typer.Option(help="Pooling ratio for built-in or custom pooling methods."),
    ] = None,
    pool_nonlinearity: Annotated[
        Optional[str],
        typer.Option(help="Pooling score nonlinearity, independent of the model activation."),
    ] = None,
    dataset: Annotated[Optional[str], typer.Option()] = None,
    model_type: Annotated[Optional[str], typer.Option(help="Model type: sum or plain.")] = None,
    log_file: Annotated[
        Optional[str],
        typer.Option(help="JSONL file path to append experiment records."),
    ] = None,
    model_config: Annotated[str, typer.Option()] = default_config_path("model.toml"),
    experiment_config: Annotated[str, typer.Option()] = default_config_path("experiment.toml"),
    tag: Annotated[
        Optional[str],
        typer.Option(help="Short tag used to group related runs in logs."),
    ] = None,
    activation_checkpoint: Annotated[
        Optional[bool],
        typer.Option(help="Use activation checkpointing to reduce GPU memory at extra compute cost."),
    ] = None,
    seed_mode: Annotated[
        Optional[str],
        typer.Option(help="Seed source mode: auto, file, or list."),
    ] = None,
    seed_base: Annotated[
        Optional[int],
        typer.Option(help="Base integer for deterministic seed generation in auto mode."),
    ] = None,
    seed_list: Annotated[
        Optional[str],
        typer.Option(help="Comma-separated seed list for exact replay, for example: 11,22,33"),
    ] = None,
    allow_duplicate_seeds: Annotated[
        Optional[bool],
        typer.Option(help="Allow duplicate seeds in file or list mode."),
    ] = None,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        model_config_data = toml.load(model_config)
        experiment_config_data = toml.load(experiment_config)
        experiment_section = experiment_config_data.get("experiment", {})
        (
            resolved_seed_mode,
            resolved_seed_base,
            resolved_allow_duplicates,
            resolved_seed_list,
        ) = resolve_seed_options(
            seed_mode=seed_mode,
            seed_base=seed_base,
            seed_list=seed_list,
            allow_duplicate_seeds=allow_duplicate_seeds,
            experiment_section=experiment_section,
        )

        spec = build_cli_spec(
            model_config=model_config_data,
            experiment_config=experiment_config_data,
            pool=pool,
            pool_ratio=pool_ratio,
            pool_nonlinearity=pool_nonlinearity,
            activation_checkpoint=activation_checkpoint,
            dataset_name=dataset,
            model_type=model_type,
            tag=tag,
            log_file=log_file,
            seed_mode=resolved_seed_mode,
            seed_base=resolved_seed_base,
            seed_list=resolved_seed_list,
            allow_duplicate_seeds=resolved_allow_duplicates,
        )

        payload = execute_train_request(spec, emit_text=output_format == "text")

        if output_format == "json":
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("train_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
