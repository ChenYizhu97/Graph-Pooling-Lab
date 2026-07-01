import toml
import typer
from typing_extensions import Annotated, Optional

from gplab.cli.options import resolve_seed_options
from gplab.cli.request import build_cli_request
from gplab.experiment.train_result import execute_train_request
from gplab.paths import default_config_path
from gplab.cli.output import (
    build_error_payload,
    emit_json,
    redirect_stdout_for_json,
    validate_output_format,
)

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
    model_variant: Annotated[Optional[str], typer.Option(help="Model variant: sum or plain.")] = None,
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
        typer.Option(help="Seed source mode: auto or list."),
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
        typer.Option(help="Allow duplicate seeds in list mode."),
    ] = None,
    split_train: Annotated[
        Optional[float],
        typer.Option(help="Training split ratio."),
    ] = None,
    split_val: Annotated[
        Optional[float],
        typer.Option(help="Validation split ratio."),
    ] = None,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    json_output = output_format == "json"
    try:
        with redirect_stdout_for_json(json_output):
            model_config_data = toml.load(model_config)
            experiment_config_data = toml.load(experiment_config)
            training_section = experiment_config_data.get("training", {})
            seeds_section = training_section.get("seeds", {})
            (
                resolved_seed_mode,
                resolved_seed_base,
                resolved_allow_duplicates,
                resolved_seed_values,
            ) = resolve_seed_options(
                seed_mode=seed_mode,
                seed_base=seed_base,
                seed_list=seed_list,
                allow_duplicate_seeds=allow_duplicate_seeds,
                seeds_section=seeds_section,
            )

            request = build_cli_request(
                model_config=model_config_data,
                training_config=experiment_config_data,
                execution_config=experiment_config_data,
                pool=pool,
                pool_ratio=pool_ratio,
                pool_nonlinearity=pool_nonlinearity,
                activation_checkpoint=activation_checkpoint,
                dataset_name=dataset,
                model_variant=model_variant,
                tag=tag,
                log_file=log_file,
                seed_mode=resolved_seed_mode,
                seed_base=resolved_seed_base,
                seed_values=resolved_seed_values,
                allow_duplicate_seeds=resolved_allow_duplicates,
                split_train=split_train,
                split_val=split_val,
            )

            payload = execute_train_request(
                request,
                emit_text=output_format == "text",
                context={
                    "source": "cli_options",
                    "case_id": request.case_id,
                    "model_config": model_config,
                    "experiment_config": experiment_config,
                },
            )

        if json_output:
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if json_output:
            emit_json(build_error_payload("train_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
