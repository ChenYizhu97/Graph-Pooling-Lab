from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.nn import summary
from tqdm import tqdm

from gplab.data.dataset import load_dataset, split_dataset
from gplab.benchmark.case import TrainingConfig
from gplab.benchmark.plan import RunPlan
from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.record import build_record
from gplab.experiment.reproducibility import (
    configure_runtime_threads,
    generate_loader,
    set_np_and_torch,
)
from gplab.model.classifier_plain import GraphClassifierPlain
from gplab.model.classifier_sum import GraphClassifierSum
from gplab.runtime import build_runtime_meta, console_separator, print_experiment_info
from gplab.train_loop import evaluate_epoch, train_epoch


@dataclass
class PreparedRun:
    request: BenchmarkRequest
    dataset: object
    dataset_profile: dict
    run_plan: RunPlan
    runtime: dict
    device: torch.device


def _profile_dataset(dataset) -> dict:
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
    dataset = load_dataset(request.case.dataset)
    dataset_profile = _profile_dataset(dataset)
    return PreparedRun(
        request=request,
        dataset=dataset,
        dataset_profile=dataset_profile,
        run_plan=RunPlan.build(request.case, dataset_profile["num_graphs"]),
        runtime=runtime,
        device=device,
    )


def _build_model(
    prepared: PreparedRun,
    device: torch.device,
):
    case = prepared.request.case
    execution = prepared.request.execution
    model_class = GraphClassifierPlain if case.model.variant == "plain" else GraphClassifierSum
    return model_class(
        prepared.dataset_profile["num_node_features"],
        prepared.dataset_profile["num_classes"],
        pool_method=case.pool.name,
        ratio=case.pool.ratio,
        pool_nonlinearity=case.pool.nonlinearity,
        config=case.model,
        avg_node_num=prepared.dataset_profile["avg_node_num"],
        activation_checkpoint=execution.activation_checkpoint,
    ).to(device)


def _execute_single_run(
    model,
    dataset,
    run_idx: int,
    run_seed: int,
    run_split: dict,
    train: TrainingConfig,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict:
    set_np_and_torch(run_seed)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, run_split)

    train_loader = generate_loader(train_dataset, train.batch_size, shuffle=True, seed=run_seed)
    val_loader = generate_loader(val_dataset, train.batch_size, shuffle=False, seed=run_seed)
    test_loader = generate_loader(test_dataset, train.batch_size, shuffle=False, seed=run_seed)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=train.lr)
    loss_fn = F.nll_loss

    best_val_loss = np.inf
    best_val_auxiliary_loss = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    loop = tqdm(range(1, train.epochs + 1), disable=not show_progress)
    for epoch in loop:
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        validation = evaluate_epoch(model, val_loader, loss_fn, device)

        if validation.classification_loss < best_val_loss:
            test = evaluate_epoch(model, test_loader, loss_fn, device)
            best_val_loss = validation.classification_loss
            best_val_auxiliary_loss = validation.auxiliary_loss
            best_test_acc = test.accuracy
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        loop.set_description(f"Run [{run_idx}/{train.runs}]-Epoch [{epoch}/{train.epochs}]")
        loop.set_postfix(
            best_epoch=best_epoch,
            best_test_acc=best_test_acc,
            best_val_loss=best_val_loss,
        )
        if stale_epochs > train.patience:
            break

    return {
        "run": run_idx,
        "seed": run_seed,
        "split_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_auxiliary_loss": best_val_auxiliary_loss,
        "best_test_acc": best_test_acc,
    }


def run_experiment(request: BenchmarkRequest, *, emit_text: bool = True) -> dict:
    configure_runtime_threads()
    set_np_and_torch(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime_meta(device)

    if emit_text:
        print_experiment_info(request.case, request.execution, device)

    prepared = prepare_run(request, device, runtime)
    model = _build_model(prepared, device)
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

    record = build_record(
        request.case,
        execution=request.execution,
        run_plan=prepared.run_plan,
        runtime=prepared.runtime,
        run_records=run_records,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record
