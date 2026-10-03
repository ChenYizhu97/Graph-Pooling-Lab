"""Execute seeded runs, select validation checkpoints, and build experiment records."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.data import Dataset
from torch_geometric.nn import summary
from tqdm import tqdm

from gplab.benchmark.compatibility import (
    resolve_dataset_connectivity_type,
    validate_pool_compatibility,
)
from gplab.benchmark.config import ExperimentConfig, TrainingConfig
from gplab.benchmark.execution import ExecutionOptions
from gplab.benchmark.runs import RunSpec, resolve_runs
from gplab.data.dataset import load_dataset, split_dataset
from gplab.environment import collect_environment_info, console_separator, print_experiment_info
from gplab.experiment.measurements import capture_structural_statistics, count_trainable_parameters
from gplab.experiment.record import build_result
from gplab.experiment.reproducibility import (
    build_loader,
    configure_runtime_threads,
    seed_everything,
)
from gplab.model import GraphClassifier
from gplab.train_loop import evaluate_epoch, train_epoch


@dataclass
class PreparedExperiment:
    """Loaded data and validated repetitions, ready for model construction and training."""
    config: ExperimentConfig
    dataset: Dataset
    dataset_stats: dict
    runs: tuple[RunSpec, ...]


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


def prepare_experiment(
    config: ExperimentConfig,
    fixed_runs: tuple[RunSpec, ...] | None = None,
) -> PreparedExperiment:
    """Load data, check pool connectivity, and resolve or validate all requested runs."""
    dataset = load_dataset(config.dataset)
    dataset_stats = _summarize_dataset(dataset)
    validate_pool_compatibility(
        dataset_type=resolve_dataset_connectivity_type(dataset),
        pool_name=config.pool.name,
        pre_conv=config.model.pre_conv,
        post_conv=config.model.post_conv,
    )
    return PreparedExperiment(
        config=config, dataset=dataset, dataset_stats=dataset_stats,
        runs=resolve_runs(config.training, len(dataset), fixed_runs),
    )


def _build_model(
    prepared: PreparedExperiment, execution: ExecutionOptions, device: torch.device,
) -> GraphClassifier:
    config = prepared.config
    return GraphClassifier(
        prepared.dataset_stats["num_node_features"],
        prepared.dataset_stats["num_classes"],
        pool_method=config.pool.name,
        ratio=config.pool.ratio,
        pool_nonlinearity=config.pool.nonlinearity,
        config=config.model,
        avg_node_num=prepared.dataset_stats["avg_node_num"],
        activation_checkpoint=execution.activation_checkpoint,
    ).to(device)


def execute_run(
    model,
    dataset,
    run_idx: int,
    run: RunSpec,
    train: TrainingConfig,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict:
    """Train from a seeded reset, restore minimum validation loss, then test exactly once."""
    seed_everything(run.seed)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, run.split.to_mapping())

    train_loader = build_loader(train_dataset, train.batch_size, shuffle=True, seed=run.seed)
    val_loader = build_loader(val_dataset, train.batch_size, shuffle=False, seed=run.seed)
    test_loader = build_loader(test_dataset, train.batch_size, shuffle=False, seed=run.seed)

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
        **run.to_mapping(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_auxiliary_loss": best_val_auxiliary_loss,
        "test_acc": test.accuracy,
        "training_wall_time_seconds": training_wall_time_seconds,
        "epochs_trained": epochs_trained,
        "peak_training_cuda_allocated_bytes": peak_training_cuda_allocated_bytes,
        "structural_stats": structural_statistics.to_mapping(),
    }


def run_experiment(
    config: ExperimentConfig,
    execution: ExecutionOptions,
    *,
    fixed_runs: tuple[RunSpec, ...] | None = None,
    emit_text: bool = True,
) -> dict:
    """Prepare and train an experiment; return measurements for the job boundary to save.

    The model is reused with a seeded reset for each run. Logging, tags, replay
    provenance, and final record identity belong to execute_job, not training.
    """
    configure_runtime_threads()
    seed_everything(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    environment = collect_environment_info(device)
    prepared = prepare_experiment(config, fixed_runs)
    model = _build_model(prepared, execution, device)
    if emit_text:
        print_experiment_info(config, execution, device)
        rprint(summary(model, data=prepared.dataset[0].to(device), leaf_module=None, max_depth=5))

    run_results = []
    for run_idx, run in enumerate(prepared.runs, start=1):
        run_results.append(execute_run(
            model, prepared.dataset, run_idx=run_idx, run=run,
            train=config.training, device=device, show_progress=emit_text,
        ))
        if emit_text and run_idx != config.training.runs:
            rprint(console_separator("-"))

    # Training materializes lazy parameters, so count them only after execution.
    result = build_result(run_results, trainable_parameters=count_trainable_parameters(model))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"environment": environment, "result": result}
