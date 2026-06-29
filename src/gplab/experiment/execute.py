import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.nn import summary
from tqdm import tqdm

from gplab.data.dataset import build_split_indices, load_dataset, split_dataset
from gplab.experiment.record import build_record
from gplab.experiment.reproducibility import (
    configure_runtime_threads,
    generate_loader,
    resolve_seeds,
    set_np_and_torch,
)
from gplab.experiment.spec import ExperimentSpec, TrainSpec
from gplab.model.classifier_plain import GraphClassifierPlain
from gplab.model.classifier_sum import GraphClassifierSum
from gplab.runtime import build_runtime_meta, console_separator, print_experiment_info
from gplab.train_loop import evaluate_epoch, train_epoch


def _build_model(
    spec: ExperimentSpec,
    dataset,
    avg_node_num: float,
    device: torch.device,
):
    model_class = GraphClassifierPlain if spec.model.variant == "plain" else GraphClassifierSum
    return model_class(
        dataset.num_node_features,
        dataset.num_classes,
        pool_method=spec.pool.name,
        ratio=spec.pool.ratio,
        pool_nonlinearity=spec.pool.nonlinearity,
        config=spec.model,
        avg_node_num=avg_node_num,
        activation_checkpoint=spec.train.activation_checkpoint,
    ).to(device)


def _resolve_run_plan(spec: ExperimentSpec, dataset_size: int) -> tuple[list[int], list[dict]]:
    train = spec.train
    seeds = resolve_seeds(
        runs=train.runs,
        seed_mode=train.seed_mode,
        seeds_path=train.seeds_path,
        seed_base=train.seed_base,
        seed_list=None if train.seed_list is None else list(train.seed_list),
        allow_duplicate_seeds=train.allow_duplicate_seeds,
    )
    splits = [
        build_split_indices(
            dataset_size,
            seed=seed,
            train_ratio=train.train_ratio,
            val_ratio=train.val_ratio,
        )
        for seed in seeds
    ]
    return seeds, splits


def _execute_single_run(
    model,
    dataset,
    run_idx: int,
    run_seed: int,
    run_split: dict,
    train: TrainSpec,
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


def run_experiment(spec: ExperimentSpec, *, emit_text: bool = True) -> dict:
    configure_runtime_threads()
    set_np_and_torch(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime_meta(device)

    if emit_text:
        print_experiment_info(spec, device)

    dataset = load_dataset(spec.dataset)
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")

    avg_node_num = sum(int(graph.num_nodes) for graph in dataset) / len(dataset)
    model = _build_model(spec, dataset, avg_node_num, device)
    if emit_text:
        rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))

    seeds, split_indices = _resolve_run_plan(spec, len(dataset))
    run_records = []
    for run_idx, (run_seed, run_split) in enumerate(zip(seeds, split_indices), start=1):
        run_records.append(
            _execute_single_run(
                model,
                dataset,
                run_idx=run_idx,
                run_seed=run_seed,
                run_split=run_split,
                train=spec.train,
                device=device,
                show_progress=emit_text,
            )
        )
        if emit_text and run_idx != spec.train.runs:
            rprint(console_separator("-"))

    record = build_record(
        spec,
        resolved_seeds=seeds,
        runtime=runtime,
        run_records=run_records,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record
