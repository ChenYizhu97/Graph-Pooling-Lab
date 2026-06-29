from typing import Optional

from gplab.experiment.spec import ExperimentSpec, ModelSpec, PoolSpec, TrainSpec


def build_cli_spec(
    *,
    model_config: dict,
    experiment_config: dict,
    pool: Optional[str],
    pool_ratio: Optional[float],
    pool_nonlinearity: Optional[str],
    activation_checkpoint: Optional[bool],
    dataset_name: Optional[str],
    model_type: Optional[str],
    tag: Optional[str],
    log_file: Optional[str],
    seed_mode: str,
    seed_base: int,
    seed_list: Optional[list[int]],
    allow_duplicate_seeds: bool,
) -> ExperimentSpec:
    if "model" not in model_config:
        raise ValueError("Missing [model] section in model config.")
    if "experiment" not in experiment_config:
        raise ValueError("Missing [experiment] section in experiment config.")

    model_section = dict(model_config["model"])
    model_section["variant"] = model_type or "sum"

    train_section = dict(experiment_config["experiment"])
    train_section.update(
        {
            "seed_mode": seed_mode,
            "seed_base": seed_base,
            "seed_list": seed_list,
            "allow_duplicate_seeds": allow_duplicate_seeds,
            "activation_checkpoint": bool(
                activation_checkpoint
                if activation_checkpoint is not None
                else train_section.get("activation_checkpoint", False)
            ),
        }
    )

    return ExperimentSpec(
        dataset=dataset_name or "PROTEINS",
        pool=PoolSpec(
            name=pool or "nopool",
            ratio=float(pool_ratio if pool_ratio is not None else 0.5),
            nonlinearity=pool_nonlinearity or "tanh",
        ),
        model=ModelSpec.from_mapping(model_section),
        train=TrainSpec.from_mapping(
            train_section,
            seeds_path=train_section.get("seeds"),
        ),
        log_file=log_file,
        tag=tag,
    )


def build_job_spec(job: dict) -> ExperimentSpec:
    return ExperimentSpec.from_job(job)
