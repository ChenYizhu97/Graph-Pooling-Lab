import numpy as np

from gplab.experiment.identity import attach_record_id, compute_benchmark_key, require_record_id
from gplab.experiment.spec import ExperimentSpec


def build_spec(spec: ExperimentSpec, resolved_seeds: list[int]) -> dict:
    return spec.to_record_spec(resolved_seeds)


def build_result(run_records: list[dict]) -> dict:
    if not run_records:
        raise ValueError("Cannot build result from an empty run record list.")

    test_acc = [float(run["best_test_acc"]) for run in run_records]
    compact_runs = [
        {
            "seed": int(run["seed"]),
            "best_epoch": int(run["best_epoch"]),
            "best_val_loss": float(run["best_val_loss"]),
            "best_val_auxiliary_loss": float(run["best_val_auxiliary_loss"]),
            "best_test_acc": float(run["best_test_acc"]),
        }
        for run in run_records
    ]
    return {
        "mean": float(np.mean(test_acc)),
        "std": float(np.std(test_acc)),
        "runs": compact_runs,
    }


def build_record(
    spec: ExperimentSpec,
    *,
    resolved_seeds: list[int],
    runtime: dict,
    run_records: list[dict],
) -> dict:
    record = {
        "spec": build_spec(spec, resolved_seeds),
        "runtime": runtime,
        "result": build_result(run_records),
    }
    if spec.tag is not None:
        record["tag"] = spec.tag
    return attach_record_id(record)


def summarize_record(record: dict) -> dict:
    ensured = require_record_id(record)
    runs = ensured["result"]["runs"]
    test_acc = [float(run["best_test_acc"]) for run in runs]
    val_loss = [float(run["best_val_loss"]) for run in runs]
    val_auxiliary_loss = [
        float(run.get("best_val_auxiliary_loss", 0.0))
        for run in runs
    ]
    epochs = [int(run["best_epoch"]) for run in runs]

    corr = None
    if len(runs) >= 2 and np.std(val_loss) != 0 and np.std(test_acc) != 0:
        corr = float(np.corrcoef(val_loss, test_acc)[0, 1])

    summary = {
        "record_id": ensured["record_id"],
        "benchmark_key": compute_benchmark_key(ensured),
        "dataset": ensured["spec"]["dataset"],
        "pool": ensured["spec"]["pool"]["name"],
        "pool_ratio": ensured["spec"]["pool"]["ratio"],
        "pool_nonlinearity": ensured["spec"]["pool"].get("nonlinearity", "tanh"),
        "activation_checkpoint": bool(ensured["spec"]["train"]["activation_checkpoint"]),
        "model_type": ensured["spec"]["model"]["variant"],
        "runs": len(runs),
        "mean": float(ensured["result"]["mean"]),
        "std": float(ensured["result"]["std"]),
        "avg_best_epoch": float(np.mean(epochs)),
        "avg_val_loss": float(np.mean(val_loss)),
        "avg_val_auxiliary_loss": float(np.mean(val_auxiliary_loss)),
        "best_test_acc": float(max(test_acc)),
        "worst_test_acc": float(min(test_acc)),
        "val_loss_test_acc_corr": corr,
    }
    if "tag" in ensured:
        summary["tag"] = ensured["tag"]
    return summary
