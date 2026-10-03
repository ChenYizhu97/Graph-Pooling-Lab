from typing import Any

import numpy as np

from gplab.benchmark.config import ExperimentConfig
from gplab.benchmark.identity import compute_comparison_group_key
from gplab.experiment.identity import attach_record_id, require_record_id

ExperimentRecord = dict[str, Any]


def build_result(run_results: list[dict], *, trainable_parameters: dict) -> dict:
    """Aggregate final-test accuracy using population standard deviation."""
    if not run_results:
        raise ValueError("Cannot build result from an empty run record list.")

    test_acc = [float(run["test_acc"]) for run in run_results]
    return {
        "mean": float(np.mean(test_acc)),
        "std": float(np.std(test_acc)),
        "trainable_parameters": {
            "total": int(trainable_parameters["total"]),
            "pooling_module": int(trainable_parameters["pooling_module"]),
        },
        "runs": run_results,
    }


def build_record(
    experiment: ExperimentConfig,
    *,
    environment: dict,
    result: dict,
    tag: str | None = None,
    source_record_id: str | None = None,
) -> ExperimentRecord:
    """Combine configuration, measurements, and provenance into a content-addressed record."""
    record = {
        "experiment": experiment.to_mapping(),
        "environment": environment,
        "result": result,
        "tag": tag,
        "source_record_id": source_record_id,
    }
    return attach_record_id(record)


def summarize_record(record: ExperimentRecord) -> dict:
    """Derive query metrics from completed runs without changing the stored record."""
    ensured = require_record_id(record)
    runs = ensured["result"]["runs"]
    test_acc = [float(run["test_acc"]) for run in runs]
    val_loss = [float(run["best_val_loss"]) for run in runs]
    val_auxiliary_loss = [float(run["best_val_auxiliary_loss"]) for run in runs]
    epochs = [int(run["best_epoch"]) for run in runs]

    # Correlation is undefined with fewer than two runs or zero variance;
    # expose null rather than a misleading zero or a non-portable JSON NaN.
    corr = None
    if len(runs) >= 2 and np.std(val_loss) != 0 and np.std(test_acc) != 0:
        corr = float(np.corrcoef(val_loss, test_acc)[0, 1])

    summary = {
        "record_id": ensured["record_id"],
        "comparison_group_key": compute_comparison_group_key(ensured),
        "dataset": ensured["experiment"]["dataset"],
        "pool": ensured["experiment"]["pool"]["name"],
        "pool_ratio": ensured["experiment"]["pool"]["ratio"],
        "pool_nonlinearity": ensured["experiment"]["pool"]["nonlinearity"],
        "activation_checkpoint": bool(ensured["experiment"]["training"]["activation_checkpoint"]),
        "model_variant": ensured["experiment"]["model"]["variant"],
        "runs": len(runs),
        "mean": float(ensured["result"]["mean"]),
        "std": float(ensured["result"]["std"]),
        "avg_best_epoch": float(np.mean(epochs)),
        "avg_val_loss": float(np.mean(val_loss)),
        "avg_val_auxiliary_loss": float(np.mean(val_auxiliary_loss)),
        "max_test_acc": float(max(test_acc)),
        "min_test_acc": float(min(test_acc)),
        "val_loss_test_acc_corr": corr,
    }
    if ensured["tag"] is not None:
        summary["tag"] = ensured["tag"]
    return summary
