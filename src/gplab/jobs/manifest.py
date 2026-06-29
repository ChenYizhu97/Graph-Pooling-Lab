from copy import deepcopy
from typing import Optional

from .defaults import AUTOMATION_EXECUTION_DEFAULTS, AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAINING_DEFAULTS
from .schema import compute_train_job_case_id, normalize_train_job


def _apply_training_overrides(training_block: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if key == "split_train":
            training_block["split"]["train"] = value
        elif key == "split_val":
            training_block["split"]["val"] = value
        elif key == "seed_mode":
            training_block["seeds"]["mode"] = value
        elif key == "seed_base":
            training_block["seeds"]["base"] = value
        elif key == "seed_list":
            training_block["seeds"]["values"] = value
        elif key == "allow_duplicate_seeds":
            training_block["seeds"]["allow_duplicates"] = value
        else:
            training_block[key] = value


def build_case_manifest(
    *,
    pools: list[str],
    datasets: list[str],
    model_variants: list[str],
    pool_ratio: float,
    pool_nonlinearity: str = "tanh",
    activation_checkpoint: bool = False,
    tag_prefix: Optional[str] = None,
    training_overrides: Optional[dict] = None,
    log_file: Optional[str] = None,
) -> list[dict]:
    manifest = []
    for dataset in datasets:
        for pool in pools:
            for model_variant in model_variants:
                model_block = deepcopy(AUTOMATION_MODEL_DEFAULTS)
                model_block["variant"] = model_variant

                training_block = deepcopy(AUTOMATION_TRAINING_DEFAULTS)
                if training_overrides:
                    _apply_training_overrides(training_block, deepcopy(training_overrides))

                execution_block = deepcopy(AUTOMATION_EXECUTION_DEFAULTS)
                execution_block["activation_checkpoint"] = activation_checkpoint
                execution_block["log_file"] = log_file
                execution_block["tag"] = (
                    f"{tag_prefix}_{pool}_{dataset}_{model_variant}" if tag_prefix else None
                )

                job = normalize_train_job(
                    {
                        "case": {
                            "dataset": dataset,
                            "pool": {
                                "name": pool,
                                "ratio": pool_ratio,
                                "nonlinearity": pool_nonlinearity,
                            },
                            "model": model_block,
                            "training": training_block,
                        },
                        "execution": execution_block,
                    }
                )
                manifest.append(
                    {
                        "case_id": compute_train_job_case_id(job),
                        "dataset": dataset,
                        "pool": pool,
                        "pool_ratio": pool_ratio,
                        "pool_nonlinearity": pool_nonlinearity,
                        "activation_checkpoint": activation_checkpoint,
                        "model_variant": model_variant,
                        "job": job,
                    }
                )
    return manifest
