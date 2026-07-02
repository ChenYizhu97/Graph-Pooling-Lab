from typing import Any

import numpy as np

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.comparison import compute_record_benchmark_key
from gplab.benchmark.execution import ExecutionOptions
from gplab.benchmark.plan import RunPlan
from gplab.experiment.identity import attach_record_id, require_record_id


ExperimentRecord = dict[str, Any]


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
    case: BenchmarkCase,
    *,
    execution: ExecutionOptions,
    run_plan: RunPlan,
    runtime: dict,
    run_records: list[dict],
) -> ExperimentRecord:
    record = {
        "case": case.to_mapping(),
        "execution": execution.to_mapping(),
        "run_plan": run_plan.to_mapping(),
        "runtime": runtime,
        "result": build_result(run_records),
    }
    return attach_record_id(record)


def summarize_record(record: ExperimentRecord) -> dict:
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
        "case_id": ensured["run_plan"]["case_id"],
        "benchmark_key": compute_record_benchmark_key(ensured),
        "dataset": ensured["case"]["dataset"],
        "pool": ensured["case"]["pool"]["name"],
        "pool_ratio": ensured["case"]["pool"]["ratio"],
        "pool_nonlinearity": ensured["case"]["pool"]["nonlinearity"],
        "activation_checkpoint": bool(ensured["execution"]["activation_checkpoint"]),
        "model_variant": ensured["case"]["model"]["variant"],
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
    if ensured["execution"].get("tag") is not None:
        summary["tag"] = ensured["execution"]["tag"]
    return summary
