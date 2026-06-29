import typer
from typing_extensions import Annotated, Optional

from gplab.cli.options import parse_csv_list, parse_seed_list
from gplab.cli.output import build_error_payload, emit_json, validate_output_format
from gplab.jobs import build_case_manifest

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    pools: Annotated[str, typer.Option(help="Comma-separated pool list.")] = "sagpool,diffpool",
    datasets: Annotated[str, typer.Option(help="Comma-separated dataset list.")] = "MUTAG,PROTEINS",
    model_variants: Annotated[str, typer.Option(help="Comma-separated model variants.")] = "sum",
    pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for every case.")] = 0.5,
    pool_nonlinearity: Annotated[str, typer.Option(help="Pooling score nonlinearity.")] = "tanh",
    activation_checkpoint: Annotated[bool, typer.Option(help="Use activation checkpointing to reduce GPU memory at extra compute cost.")] = False,
    runs: Annotated[Optional[int], typer.Option(help="Optional run count override.")] = None,
    epochs: Annotated[Optional[int], typer.Option(help="Optional epoch override.")] = None,
    patience: Annotated[Optional[int], typer.Option(help="Optional patience override.")] = None,
    lr: Annotated[Optional[float], typer.Option(help="Optional learning-rate override.")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="Optional batch-size override.")] = None,
    split_train: Annotated[Optional[float], typer.Option(help="Optional train split override.")] = None,
    split_val: Annotated[Optional[float], typer.Option(help="Optional validation split override.")] = None,
    seed_mode: Annotated[Optional[str], typer.Option(help="Optional seed mode override.")] = None,
    seed_base: Annotated[Optional[int], typer.Option(help="Optional seed base override.")] = None,
    seed_list: Annotated[Optional[str], typer.Option(help="Optional comma-separated seed list.")] = None,
    allow_duplicate_seeds: Annotated[Optional[bool], typer.Option(help="Optional duplicate-seed override.")] = None,
    log_file: Annotated[Optional[str], typer.Option(help="Optional log file for generated jobs.")] = None,
    tag_prefix: Annotated[Optional[str], typer.Option(help="Optional tag prefix for generated cases.")] = None,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "json",
):
    output_format = validate_output_format(output_format)
    try:
        parsed_seed_list = parse_seed_list(seed_list)
        training_overrides = {}
        for key, value in (
            ("runs", runs),
            ("epochs", epochs),
            ("patience", patience),
            ("lr", lr),
            ("batch_size", batch_size),
            ("split_train", split_train),
            ("split_val", split_val),
            ("seed_mode", seed_mode),
            ("seed_base", seed_base),
            ("allow_duplicate_seeds", allow_duplicate_seeds),
        ):
            if value is not None:
                training_overrides[key] = value
        if parsed_seed_list is not None:
            training_overrides["seed_list"] = parsed_seed_list
            training_overrides["seed_mode"] = "list"
        cases = build_case_manifest(
            pools=parse_csv_list(pools),
            datasets=parse_csv_list(datasets),
            model_variants=parse_csv_list(model_variants),
            pool_ratio=pool_ratio,
            pool_nonlinearity=pool_nonlinearity,
            activation_checkpoint=activation_checkpoint,
            tag_prefix=tag_prefix,
            training_overrides=training_overrides or None,
            log_file=log_file,
        )
        payload = {
            "ok": True,
            "kind": "case_manifest",
            "cases": cases,
            "summary": {
                "total": len(cases),
            },
        }
        if output_format == "json":
            emit_json(payload)
            return
        print(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("expand_cases_error", exc))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
