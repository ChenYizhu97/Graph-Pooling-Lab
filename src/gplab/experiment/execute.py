from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from rich import print as rprint
from torch_geometric.nn import summary
from torcheval.metrics import Mean, MulticlassAccuracy
from tqdm import tqdm

from gplab.experiment.record import build_record
from gplab.model.classifier_plain import GraphClassifierPlain
from gplab.model.classifier_sum import GraphClassifierSum
from gplab.train_loop import evaluate_epoch, train_epoch
from gplab.data.dataset import build_split_indices, load_dataset, split_dataset
from gplab.runtime import build_runtime_meta, print_expr_info, sep_c
from gplab.experiment.reproducibility import (
    configure_runtime_threads,
    generate_loader,
    resolve_seeds,
    set_np_and_torch,
)


def _build_model(conf: dict, dataset, avg_node_num: float, device: torch.device):
    model_type = conf["model"]["variant"]
    model_class = GraphClassifierPlain if model_type == "plain" else GraphClassifierSum
    return model_class(
        dataset.num_node_features,
        dataset.num_classes,
        pool_method=conf["pool"]["method"],
        ratio=conf["pool"]["ratio"],
        config=conf["model"],
        avg_node_num=avg_node_num,
        activation_checkpoint=conf["experiment"]["activation_checkpoint"],
    ).to(device)


def _prepare_split_metadata(conf: dict, dataset_size: int) -> list[dict]:
    expr_conf = conf["experiment"]
    seeds = resolve_seeds(
        runs=expr_conf["runs"],
        seed_mode=expr_conf["seed_mode"],
        seeds_path=expr_conf.get("seeds"),
        seed_base=expr_conf["seed_base"] if expr_conf["seed_base"] is not None else 20260320,
        seed_list=expr_conf.get("seed_list"),
        allow_duplicate_seeds=expr_conf["allow_duplicate_seeds"],
    )
    expr_conf["seeds"] = seeds

    split_indices_all = [
        build_split_indices(
            dataset_size,
            seed=seed,
            train_ratio=expr_conf["train_ratio"],
            val_ratio=expr_conf["val_ratio"],
        )
        for seed in seeds
    ]
    return split_indices_all


def _execute_single_run(
    model,
    dataset,
    run_idx: int,
    run_seed: int,
    run_split: dict,
    expr_conf: dict,
    metrics: dict,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict:
    set_np_and_torch(run_seed)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_indices=run_split)

    train_loader = generate_loader(train_dataset, expr_conf["batch_size"], shuffle=True, seed=run_seed)
    val_loader = generate_loader(val_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)
    test_loader = generate_loader(test_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=expr_conf["lr"])
    loss_fn = F.nll_loss

    best_val_loss = np.inf
    best_test_acc = 0.0
    best_epoch = 1

    loop = tqdm(range(1, expr_conf["epochs"] + 1), disable=not show_progress)
    for epoch in loop:
        train_epoch(model, train_loader, optimizer, loss_fn, metrics, device)
        _, val_loss = evaluate_epoch(model, val_loader, loss_fn, metrics, device)
        test_acc, _ = evaluate_epoch(model, test_loader, loss_fn, metrics, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
            best_epoch = epoch

        if epoch > best_epoch + expr_conf["patience"]:
            break

        loop.set_description(f"Run [{run_idx}/{expr_conf['runs']}]-Epoch [{epoch}/{expr_conf['epochs']}]")
        loop.set_postfix(best_epoch=best_epoch, best_test_acc=best_test_acc, best_val_loss=best_val_loss)

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
        "best_test_acc": best_test_acc,
    }


def run_experiment(conf: dict, *, emit_text: bool = True) -> dict:
    working_conf = deepcopy(conf)
    configure_runtime_threads()
    set_np_and_torch(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime_meta(device)
    expr_conf = working_conf["experiment"]

    if emit_text:
        print_expr_info(working_conf, device)

    dataset = load_dataset(working_conf["dataset"])
    if dataset is None:
        raise ValueError(f"Failed to load dataset '{working_conf['dataset']}'.")
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")

    avg_node_num = dataset._data.num_nodes // len(dataset)
    model = _build_model(working_conf, dataset, avg_node_num, device)
    if emit_text:
        rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))

    split_indices_all = _prepare_split_metadata(working_conf, len(dataset))
    metrics = {
        "loss": Mean(device=device),
        "acc": MulticlassAccuracy(average="micro", device=device, num_classes=dataset.num_classes),
    }

    run_records = []

    for run_idx in range(1, expr_conf["runs"] + 1):
        run_record = _execute_single_run(
            model,
            dataset,
            run_idx=run_idx,
            run_seed=expr_conf["seeds"][run_idx - 1],
            run_split=split_indices_all[run_idx - 1],
            expr_conf=expr_conf,
            metrics=metrics,
            device=device,
            show_progress=emit_text,
        )
        if emit_text and run_idx != expr_conf["runs"]:
            rprint(sep_c("-"))

        run_records.append(run_record)

    return build_record(working_conf, runtime=runtime, run_records=run_records)
