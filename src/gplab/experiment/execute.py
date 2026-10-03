"""Execute seeded runs, select validation checkpoints, and build experiment records."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.nn import summary
from tqdm import tqdm

from gplab.benchmark.case import TrainingConfig
from gplab.benchmark.compatibility import (
    resolve_dataset_connectivity_type,
    validate_pool_compatibility,
)
from gplab.benchmark.plan import RunPlan, SplitIndices
from gplab.benchmark.request import BenchmarkRequest
from gplab.data.dataset import load_dataset, split_dataset
from gplab.experiment.measurements import capture_structural_statistics, count_trainable_parameters
from gplab.experiment.record import build_record
from gplab.experiment.reproducibility import (
    build_loader,
    configure_runtime_threads,
    seed_everything,
)
from gplab.model import GraphClassifier
from gplab.runtime import build_runtime_meta, console_separator, print_experiment_info
from gplab.train_loop import evaluate_epoch, train_epoch


@dataclass
class PreparedRun:
    """Loaded dataset, validated run plan, and execution metadata for one request."""
    request: BenchmarkRequest
    dataset: object
    dataset_stats: dict
    run_plan: RunPlan
    runtime: dict
    device: torch.device


def _summarize_dataset(dataset) -> dict:
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")
    avg_node_num = sum(int(graph.num_nodes) for graph in dataset) / len(dataset)
    return {
        "num_graphs": len(dataset),
        "num_node_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "avg_node_num": avg_node_num,
    }


def prepare_run(request: BenchmarkRequest, device: torch.device, runtime: dict) -> PreparedRun:
    """Load the dataset and validate connectivity and split bounds before building a model."""
    dataset = load_dataset(request.case.dataset)
    dataset_stats = _summarize_dataset(dataset)
    dataset_type = resolve_dataset_connectivity_type(dataset)
    validate_pool_compatibility(
        dataset_type=dataset_type,
        pool_name=request.case.pool.name,
        pre_conv=request.case.model.pre_conv,
        post_conv=request.case.model.post_conv,
    )
    run_plan = request.fixed_run_plan or RunPlan.build(request.case, dataset_stats["num_graphs"])
    run_plan.validate_for_execution(
        runs=request.case.training.runs,
        dataset_size=dataset_stats["num_graphs"],
    )
    return PreparedRun(
        request=request,
        dataset=dataset,
        dataset_stats=dataset_stats,
        run_plan=run_plan,
        runtime=runtime,
        device=device,
    )


def _build_model(prepared: PreparedRun) -> GraphClassifier:
    case = prepared.request.case
    execution = prepared.request.execution
    return GraphClassifier(
        prepared.dataset_stats["num_node_features"],
        prepared.dataset_stats["num_classes"],
        pool_method=case.pool.name,
        ratio=case.pool.ratio,
        pool_nonlinearity=case.pool.nonlinearity,
        config=case.model,
        avg_node_num=prepared.dataset_stats["avg_node_num"],
        activation_checkpoint=execution.activation_checkpoint,
    ).to(prepared.device)


def _execute_single_run(
    model,
    dataset,
    run_idx: int,
    run_seed: int,
    run_split: SplitIndices,
    train: TrainingConfig,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict:
    """Train from a seeded reset, restore minimum validation loss, then test exactly once."""
    seed_everything(run_seed)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, run_split.to_mapping())

    train_loader = build_loader(train_dataset, train.batch_size, shuffle=True, seed=run_seed)
    val_loader = build_loader(val_dataset, train.batch_size, shuffle=False, seed=run_seed)
    test_loader = build_loader(test_dataset, train.batch_size, shuffle=False, seed=run_seed)

    # The model is reused across runs. Reset it after seeding and create a new
    # optimizer so neither learned weights nor Adam state carry into the next run.
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=train.lr)
    loss_fn = F.nll_loss

    best_val_loss = np.inf
    best_val_auxiliary_loss = 0.0
    best_checkpoint = None
    best_epoch = 0
    stale_epochs = 0
    epochs_trained = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    loop = tqdm(range(1, train.epochs + 1), disable=not show_progress)
    for epoch in loop:
        epochs_trained = epoch
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        validation = evaluate_epoch(model, val_loader, loss_fn, device)

        # Pool auxiliary losses affect training, but checkpoint selection uses
        # classification loss alone so methods share the same selection criterion.
        if validation.classification_loss < best_val_loss:
            best_val_loss = validation.classification_loss
            best_val_auxiliary_loss = validation.auxiliary_loss
            # state_dict tensors share model storage; copy them so later updates
            # cannot change the checkpoint selected here.
            best_checkpoint = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        loop.set_description(f"Run [{run_idx}/{train.runs}]-Epoch [{epoch}/{train.epochs}]")
        loop.set_postfix(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        )
        # Keep the existing strict boundary: stop on patience + 1 consecutive
        # non-improving epochs; patience=0 stops at the first non-improvement.
        if stale_epochs > train.patience:
            break

    # Finish training measurements before restoring/testing the checkpoint.
    # Synchronization includes queued CUDA work; final-test hooks and evaluation
    # must not inflate the training-only memory peak or wall time.
    peak_training_cuda_allocated_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_training_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    training_wall_time_seconds = time.perf_counter() - training_started

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_checkpoint)
    with capture_structural_statistics(model) as structural_statistics:
        test = evaluate_epoch(model, test_loader, loss_fn, device)

    return {
        "seed": run_seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_auxiliary_loss": best_val_auxiliary_loss,
        "test_acc": test.accuracy,
        "training_wall_time_seconds": training_wall_time_seconds,
        "epochs_trained": epochs_trained,
        "peak_training_cuda_allocated_bytes": peak_training_cuda_allocated_bytes,
        "structural_stats": structural_statistics.to_mapping(),
    }


def run_experiment(request: BenchmarkRequest, *, emit_text: bool = True) -> dict:
    """Execute all planned seeds with one reusable model and return a canonical record."""
    configure_runtime_threads()
    seed_everything(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime_meta(device)

    if emit_text:
        print_experiment_info(request.case, request.execution, device)

    prepared = prepare_run(request, device, runtime)
    model = _build_model(prepared)
    if emit_text:
        rprint(summary(model, data=prepared.dataset[0].to(device), leaf_module=None, max_depth=5))

    run_records = []
    for run_idx, (run_seed, run_split) in enumerate(
        zip(prepared.run_plan.seeds, prepared.run_plan.splits),
        start=1,
    ):
        run_records.append(
            _execute_single_run(
                model,
                prepared.dataset,
                run_idx=run_idx,
                run_seed=run_seed,
                run_split=run_split,
                train=request.case.training,
                device=device,
                show_progress=emit_text,
            )
        )
        if emit_text and run_idx != request.case.training.runs:
            rprint(console_separator("-"))

    # Normal training has now materialized any lazy parameters in custom pools.
    trainable_parameters = count_trainable_parameters(model)
    record = build_record(
        request.case,
        execution=request.execution,
        run_plan=prepared.run_plan,
        runtime=prepared.runtime,
        run_records=run_records,
        trainable_parameters=trainable_parameters,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record
