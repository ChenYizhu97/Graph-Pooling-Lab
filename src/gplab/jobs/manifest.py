from copy import deepcopy
from typing import Optional

from .defaults import AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAIN_DEFAULTS
from .schema import compute_train_job_case_id, normalize_train_job


def build_case_manifest(
    *,
    pools: list[str],
    datasets: list[str],
    model_types: list[str],
    pool_ratio: float,
    activation_checkpoint: bool = False,
    tag_prefix: Optional[str] = None,
    train_overrides: Optional[dict] = None,
    log_file: Optional[str] = None,
) -> list[dict]:
    manifest = []
    for dataset in datasets:
        for pool in pools:
            for model_type in model_types:
                model_block = deepcopy(AUTOMATION_MODEL_DEFAULTS)
                model_block["variant"] = model_type

                train_block = deepcopy(AUTOMATION_TRAIN_DEFAULTS)
                if train_overrides:
                    train_block.update(deepcopy(train_overrides))
                train_block["activation_checkpoint"] = activation_checkpoint

                job = normalize_train_job(
                    {
                        "dataset": dataset,
                        "pool": {"name": pool, "ratio": pool_ratio},
                        "model": model_block,
                        "train": train_block,
                        "log_file": log_file,
                        "tag": f"{tag_prefix}_{pool}_{dataset}_{model_type}" if tag_prefix else None,
                    }
                )
                manifest.append(
                    {
                        "case_id": compute_train_job_case_id(job),
                        "dataset": dataset,
                        "pool": pool,
                        "pool_ratio": pool_ratio,
                        "activation_checkpoint": activation_checkpoint,
                        "model_type": model_type,
                        "job": job,
                    }
                )
    return manifest
