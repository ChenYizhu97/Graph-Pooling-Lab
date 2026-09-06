import copy
from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.nn import summary
from tqdm import tqdm

from gplab.data.dataset import load_dataset, split_dataset
from gplab.benchmark.case import TrainingConfig
from gplab.benchmark.plan import RunPlan, SplitIndices
from gplab.benchmark.request import BenchmarkRequest
from gplab.benchmark.comparability import resolve_dataset_connectivity_type, validate_comparability
from gplab.experiment.record import build_record
from gplab.experiment.reproducibility import (
    configure_runtime_threads,
    generate_loader,
    set_np_and_torch,
)
from gplab.model import GraphClassifier
from gplab.runtime import build_runtime_meta, console_separator, print_experiment_info
from gplab.train_loop import evaluate_epoch, train_epoch


@dataclass
class PreparedRun:
    request: BenchmarkRequest
    dataset: object
    dataset_stats: dict
    run_plan: RunPlan
    runtime: dict
    device: torch.device


@dataclass
class _StructuralStatistics:
    total_input_nodes: int = 0
    total_output_nodes: int = 0
    total_input_edges: int = 0
    total_output_edges: int = 0
    total_input_nonzero_edges: int = 0
    total_output_nonzero_edges: int = 0
    num_graphs: int = 0
    _node_retention_sum: float = 0.0

    def observe(
        self,
        *,
        input_x: torch.Tensor,
        input_edge_index: torch.Tensor,
        input_batch: torch.Tensor,
        output_x: torch.Tensor,
        output_edge_index: torch.Tensor,
        output_batch: torch.Tensor,
        input_edge_weight: torch.Tensor | None = None,
        output_edge_weight: torch.Tensor | None = None,
    ) -> None:
        if input_x.size(0) == 0:
            raise ValueError("Cannot measure pooling structure for an empty graph batch.")

        graph_count = int(input_batch.max().item()) + 1
        input_nodes_per_graph = torch.bincount(input_batch, minlength=graph_count)
        output_nodes_per_graph = torch.bincount(output_batch, minlength=graph_count)
        if output_nodes_per_graph.numel() != graph_count:
            raise ValueError("Pooled batch contains a graph index absent from the input batch.")
        if torch.any(input_nodes_per_graph == 0):
            raise ValueError("Input batch graph indices must be contiguous.")

        self.total_input_nodes += int(input_x.size(0))
        self.total_output_nodes += int(output_x.size(0))
        self.total_input_edges += int(input_edge_index.size(1))
        self.total_output_edges += int(output_edge_index.size(1))
        self.total_input_nonzero_edges += _count_nonzero_edges(input_edge_index, input_edge_weight)
        self.total_output_nonzero_edges += _count_nonzero_edges(output_edge_index, output_edge_weight)
        self.num_graphs += graph_count
        self._node_retention_sum += float(
            (
                output_nodes_per_graph.to(torch.float64)
                / input_nodes_per_graph.to(torch.float64)
            ).sum().item()
        )

    def to_mapping(self) -> dict:
        if self.num_graphs == 0:
            raise ValueError("Structural statistics observed no graphs.")
        return {
            "total_input_nodes": self.total_input_nodes,
            "total_output_nodes": self.total_output_nodes,
            "total_input_edges": self.total_input_edges,
            "total_output_edges": self.total_output_edges,
            "total_input_nonzero_edges": self.total_input_nonzero_edges,
            "total_output_nonzero_edges": self.total_output_nonzero_edges,
            "num_graphs": self.num_graphs,
            "mean_node_retention": self._node_retention_sum / self.num_graphs,
        }


def _count_nonzero_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> int:
    if edge_weight is None:
        return int(edge_index.size(1))
    return int(torch.count_nonzero(edge_weight).item())


def _argument(args: tuple, kwargs: dict, name: str, position: int, default=None):
    if name in kwargs:
        return kwargs[name]
    return args[position] if position < len(args) else default


@contextmanager
def _capture_structural_statistics(model):
    statistics = _StructuralStatistics()
    pool_module = getattr(model, "pool_module", None)

    if pool_module is None:
        def observe_identity(_module, args, kwargs):
            data = _argument(args, kwargs, "data", 0)
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = data.edge_index.new_zeros(data.x.size(0))
            statistics.observe(
                input_x=data.x,
                input_edge_index=data.edge_index,
                input_batch=batch,
                output_x=data.x,
                output_edge_index=data.edge_index,
                output_batch=batch,
                input_edge_weight=getattr(data, "edge_weight", None),
                output_edge_weight=getattr(data, "edge_weight", None),
            )

        handle = model.register_forward_pre_hook(observe_identity, with_kwargs=True)
    else:
        def observe_pool(_module, args, kwargs, output):
            statistics.observe(
                input_x=_argument(args, kwargs, "x", 0),
                input_edge_index=_argument(args, kwargs, "edge_index", 1),
                input_batch=_argument(args, kwargs, "batch", 2),
                output_x=output.x,
                output_edge_index=output.edge_index,
                output_batch=output.batch,
                input_edge_weight=_argument(args, kwargs, "edge_weight", 3),
                output_edge_weight=output.edge_weight,
            )

        handle = pool_module.register_forward_hook(observe_pool, with_kwargs=True)

    try:
        yield statistics
    finally:
        handle.remove()


def _count_trainable_parameters(model) -> dict:
    pool_module = getattr(model, "pool_module", None)
    return {
        "total": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "pooling_module": 0 if pool_module is None else sum(
            parameter.numel()
            for parameter in pool_module.parameters()
            if parameter.requires_grad
        ),
    }


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
    dataset = load_dataset(request.case.dataset)
    dataset_stats = _summarize_dataset(dataset)
    dataset_type = resolve_dataset_connectivity_type(dataset)
    validate_comparability(
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


def _build_model(
    prepared: PreparedRun,
    device: torch.device,
):
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
    ).to(device)


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
    set_np_and_torch(run_seed)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, run_split.to_mapping())

    train_loader = generate_loader(train_dataset, train.batch_size, shuffle=True, seed=run_seed)
    val_loader = generate_loader(val_dataset, train.batch_size, shuffle=False, seed=run_seed)
    test_loader = generate_loader(test_dataset, train.batch_size, shuffle=False, seed=run_seed)

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

        if validation.classification_loss < best_val_loss:
            best_val_loss = validation.classification_loss
            best_val_auxiliary_loss = validation.auxiliary_loss
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
        if stale_epochs > train.patience:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_time_seconds = time.perf_counter() - training_started

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_checkpoint)
    with _capture_structural_statistics(model) as structural_statistics:
        test = evaluate_epoch(model, test_loader, loss_fn, device)

    peak_cuda_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

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
        "best_test_acc": test.accuracy,
        "training_wall_time_seconds": training_wall_time_seconds,
        "epochs_trained": epochs_trained,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "structural_stats": structural_statistics.to_mapping(),
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

    # Normal training has now materialized any lazy parameters in custom pools.
    trainable_parameters = _count_trainable_parameters(model)
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
