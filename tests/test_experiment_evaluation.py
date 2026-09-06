from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data, InMemoryDataset

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.execution import ExecutionOptions
from gplab.benchmark.plan import SplitIndices
from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.execute import (
    _capture_structural_statistics,
    _count_trainable_parameters,
    _execute_single_run,
    run_experiment,
)
from gplab.experiment.record import build_result
from gplab.graph import ConnectivityType
from gplab.layers.pool.profiles import POOLING_PROFILES, PoolingProfile, PoolingSignature
from gplab.layers.pool.pooling_output import PoolingOutput
from gplab.model import GraphClassifier
from gplab.train_loop import EvaluationResult


def _case() -> BenchmarkCase:
    return BenchmarkCase.from_mapping({
        "dataset": "MUTAG",
        "pool": {"name": "nopool", "ratio": 0.5, "nonlinearity": "tanh"},
        "model": {
            "hidden_features": 4,
            "nonlinearity": "relu",
            "p_dropout": 0.0,
            "pre_conv": "GCN",
            "post_conv": "GCN",
            "pre_gnn": [4],
            "post_gnn": [8, 4],
            "variant": "plain",
        },
        "training": {
            "runs": 1,
            "lr": 0.001,
            "batch_size": 2,
            "patience": 0,
            "epochs": 3,
            "split": {"train": 0.5, "val": 0.25},
            "seeds": {
                "mode": "list",
                "base": 1,
                "values": [1],
                "allow_duplicates": False,
            },
        },
    })


class _SelectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.pool_module = None

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, data):
        return data.x.new_zeros((data.y.numel(), 2)), None


class _FixedPool(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)

    def forward(self, *, x, edge_index, batch, edge_weight=None):
        selected = torch.tensor([0, 1, 3], device=x.device)
        return PoolingOutput(
            x=x[selected],
            edge_index=torch.tensor([[0, 1], [1, 0]], device=x.device),
            batch=torch.tensor([0, 0, 1], device=x.device),
            edge_weight=None if edge_weight is None else edge_weight[:2],
        )


class _PooledModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.pool_module = _FixedPool()

    def forward(self, data):
        return self.pool_module(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_weight=getattr(data, "edge_weight", None),
        )


class _LazyPool(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.linear = torch.nn.LazyLinear(in_channels)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, *, x, edge_index, batch, edge_weight=None):
        return PoolingOutput(
            x=self.linear(x),
            edge_index=edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )


class ExperimentEvaluationTests(unittest.TestCase):
    def test_run_experiment_counts_lazy_pool_parameters_after_training(self):
        dataset = InMemoryDataset()
        dataset.data, dataset.slices = dataset.collate([
            Data(
                x=torch.ones(3, 2),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                y=torch.tensor([index % 2]),
            )
            for index in range(8)
        ])
        dataset.connectivity_type = ConnectivityType.BINARY
        profile = PoolingProfile(
            builder=lambda channels, _ratio, _avg_nodes, _act: _LazyPool(channels),
            signatures=(PoolingSignature(ConnectivityType.BINARY, ConnectivityType.BINARY),),
        )
        with (
            patch(
                "gplab.layers.pool.profiles.POOLING_PROFILES",
                {**POOLING_PROFILES, "lazy_test_pool": profile},
            ),
            patch("gplab.experiment.execute.load_dataset", return_value=dataset),
            patch("gplab.experiment.execute.torch.cuda.is_available", return_value=False),
        ):
            case = _case()
            case = replace(
                case,
                pool=replace(case.pool, name="lazy_test_pool"),
                training=replace(
                    case.training,
                    epochs=1,
                    runs=2,
                    seeds=replace(case.training.seeds, values=(1, 2)),
                ),
            )
            record = run_experiment(
                BenchmarkRequest(case=case, execution=ExecutionOptions()),
                emit_text=False,
            )

        self.assertEqual(record["result"]["trainable_parameters"]["pooling_module"], 20)
        self.assertGreater(record["result"]["trainable_parameters"]["total"], 20)
        self.assertEqual(len(record["result"]["runs"]), 2)

    def test_restores_validation_checkpoint_then_evaluates_test_once(self):
        model = _SelectionModel()
        training = replace(_case().training, epochs=3, patience=0)
        test_batch = Data(
            x=torch.ones(5, 2),
            edge_index=torch.tensor([[0, 1, 1, 3], [1, 0, 2, 4]]),
            batch=torch.tensor([0, 0, 0, 1, 1]),
            y=torch.tensor([0, 1]),
        )
        validation_losses = iter([3.0, 1.0, 2.0])
        evaluated_loaders = []

        def train_one_epoch(current_model, *_args):
            with torch.no_grad():
                current_model.weight.add_(1)

        def evaluate(current_model, loader, *_args):
            evaluated_loaders.append(loader)
            if loader == "val":
                return EvaluationResult(
                    accuracy=0.5,
                    classification_loss=next(validation_losses),
                    auxiliary_loss=0.25,
                )
            self.assertEqual(loader, "test")
            self.assertEqual(float(current_model.weight), 2.0)
            current_model(test_batch)
            return EvaluationResult(
                accuracy=0.75,
                classification_loss=0.5,
                auxiliary_loss=0.0,
            )

        with (
            patch(
                "gplab.experiment.execute.split_dataset",
                return_value=([0], [1], [2, 3]),
            ),
            patch(
                "gplab.experiment.execute.generate_loader",
                side_effect=["train", "val", "test"],
            ),
            patch("gplab.experiment.execute.train_epoch", side_effect=train_one_epoch),
            patch("gplab.experiment.execute.evaluate_epoch", side_effect=evaluate),
            patch("gplab.experiment.execute.time.perf_counter", side_effect=[10.0, 12.5]),
        ):
            run = _execute_single_run(
                model,
                dataset=[0, 1, 2, 3],
                run_idx=1,
                run_seed=7,
                run_split=SplitIndices(train=(0,), val=(1,), test=(2, 3)),
                train=training,
                device=torch.device("cpu"),
                show_progress=False,
            )

        self.assertEqual(evaluated_loaders, ["val", "val", "val", "test"])
        self.assertEqual(run["best_epoch"], 2)
        self.assertEqual(run["epochs_trained"], 3)
        self.assertEqual(run["best_test_acc"], 0.75)
        self.assertEqual(run["training_wall_time_seconds"], 2.5)
        self.assertIsNone(run["peak_cuda_allocated_bytes"])

    def test_structural_statistics_are_aggregate_per_graph_ratios(self):
        model = _PooledModel()
        batch = Data(
            x=torch.ones(5, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]
            ),
            batch=torch.tensor([0, 0, 0, 1, 1]),
        )

        with _capture_structural_statistics(model) as statistics:
            model(batch)

        self.assertEqual(
            statistics.to_mapping(),
            {
                "total_input_nodes": 5,
                "total_output_nodes": 3,
                "total_input_edges": 6,
                "total_output_edges": 2,
                "total_input_nonzero_edges": 6,
                "total_output_nonzero_edges": 2,
                "num_graphs": 2,
                "mean_node_retention": (2 / 3 + 1 / 2) / 2,
            },
        )
        for field in statistics.to_mapping():
            self.assertNotIn(field, PoolingOutput.__dataclass_fields__)

    def test_nonzero_edge_counts_preserve_signed_and_small_weights(self):
        batch = Data(
            x=torch.ones(5, 2),
            edge_index=torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
            batch=torch.tensor([0, 0, 0, 1, 1]),
            y=torch.tensor([0, 1]),
            edge_weight=torch.tensor([-0.5, 0.0, 1e-12, 0.0, 2.0, 0.0]),
        )
        for model, output_edges, output_nonzero in (
            (_SelectionModel(), 6, 3),
            (_PooledModel(), 2, 1),
        ):
            with self.subTest(model=type(model).__name__):
                with _capture_structural_statistics(model) as statistics:
                    model(batch)
                measured = statistics.to_mapping()
                self.assertEqual(measured["total_input_edges"], 6)
                self.assertEqual(measured["total_input_nonzero_edges"], 3)
                self.assertEqual(measured["total_output_edges"], output_edges)
                self.assertEqual(measured["total_output_nonzero_edges"], output_nonzero)

    def test_dense_edge_counts_distinguish_zero_weight_slots(self):
        for pool_name, has_edges, nonzero_edges in (
            ("mincutpool", True, 2),
            ("diffpool", False, 0),
            ("densepool", False, 0),
        ):
            with self.subTest(pool=pool_name):
                graph = Data(
                    x=torch.ones(4, 2),
                    edge_index=(
                        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
                        if has_edges else torch.empty((2, 0), dtype=torch.long)
                    ),
                    batch=torch.zeros(4, dtype=torch.long),
                )
                model = GraphClassifier(
                    2, 2, _case().model, pool_method=pool_name, ratio=0.5, avg_node_num=4,
                ).eval()
                with torch.no_grad(), _capture_structural_statistics(model) as statistics:
                    model(graph)
                measured = statistics.to_mapping()
                self.assertEqual(measured["total_output_edges"], 4)
                self.assertEqual(measured["total_output_nonzero_edges"], nonzero_edges)
                self.assertEqual(measured["total_input_edges"], 6 if has_edges else 0)
                self.assertEqual(measured["total_input_nonzero_edges"], 6 if has_edges else 0)

    def test_all_builtin_pools_emit_structural_totals(self):
        batch = Data(
            x=torch.ones(8, 2),
            edge_index=torch.tensor([
                [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
            ]),
            batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        )

        for pool_name in POOLING_PROFILES:
            with self.subTest(pool_name=pool_name):
                model = GraphClassifier(
                    n_node_features=2,
                    n_classes=2,
                    config=_case().model,
                    pool_method=pool_name,
                    ratio=0.5,
                    avg_node_num=4,
                )
                model.eval()
                with _capture_structural_statistics(model) as statistics:
                    model(batch)
                measured = statistics.to_mapping()

                self.assertEqual(measured["total_input_nodes"], 8)
                self.assertEqual(measured["total_input_edges"], 12)
                self.assertEqual(measured["total_input_nonzero_edges"], 12)
                self.assertLessEqual(measured["total_output_nonzero_edges"], measured["total_output_edges"])
                self.assertEqual(measured["num_graphs"], 2)
                self.assertGreater(measured["total_output_nodes"], 0)
                self.assertGreater(measured["mean_node_retention"], 0.0)

    def test_result_records_counts_without_per_graph_details(self):
        model = _PooledModel()
        counts = _count_trainable_parameters(model)
        self.assertEqual(counts, {"total": 11, "pooling_module": 3})

        run = {
            "seed": 7,
            "best_epoch": 2,
            "best_val_loss": 0.5,
            "best_val_auxiliary_loss": 0.0,
            "best_test_acc": 0.75,
            "training_wall_time_seconds": 1.25,
            "epochs_trained": 3,
            "peak_cuda_allocated_bytes": None,
            "structural_stats": {
                "total_input_nodes": 5,
                "total_output_nodes": 3,
                "total_input_edges": 6,
                "total_output_edges": 2,
                "total_input_nonzero_edges": 4,
                "total_output_nonzero_edges": 1,
                "num_graphs": 2,
                "mean_node_retention": (2 / 3 + 1 / 2) / 2,
            },
        }
        result = build_result([run], trainable_parameters=counts)

        self.assertEqual(result["trainable_parameters"], counts)
        self.assertEqual(result["runs"][0]["structural_stats"], run["structural_stats"])
        self.assertFalse(any(
            isinstance(value, list)
            for value in result["runs"][0]["structural_stats"].values()
        ))

    def test_cuda_peak_memory_is_reset_and_recorded(self):
        model = _SelectionModel()
        test_batch = Data(
            x=torch.ones(2, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            batch=torch.zeros(2, dtype=torch.long),
            y=torch.tensor([0]),
        )

        def train_one_epoch(current_model, *_args):
            with torch.no_grad():
                current_model.weight.add_(1)

        def evaluate(current_model, loader, *_args):
            if loader == "test":
                current_model(test_batch)
            return EvaluationResult(0.5, 1.0, 0.0)

        with (
            patch(
                "gplab.experiment.execute.split_dataset",
                return_value=([0], [1], [2]),
            ),
            patch(
                "gplab.experiment.execute.generate_loader",
                side_effect=["train", "val", "test"],
            ),
            patch("gplab.experiment.execute.train_epoch", side_effect=train_one_epoch),
            patch("gplab.experiment.execute.evaluate_epoch", side_effect=evaluate),
            patch("gplab.experiment.execute.torch.cuda.synchronize") as synchronize,
            patch("gplab.experiment.execute.torch.cuda.reset_peak_memory_stats") as reset_peak,
            patch(
                "gplab.experiment.execute.torch.cuda.max_memory_allocated",
                return_value=4096,
            ) as max_allocated,
        ):
            run = _execute_single_run(
                model,
                dataset=[0, 1, 2],
                run_idx=1,
                run_seed=7,
                run_split=SplitIndices(train=(0,), val=(1,), test=(2,)),
                train=replace(_case().training, epochs=1),
                device=torch.device("cuda:0"),
                show_progress=False,
            )

        self.assertEqual(run["peak_cuda_allocated_bytes"], 4096)
        self.assertEqual(synchronize.call_count, 2)
        reset_peak.assert_called_once_with(torch.device("cuda:0"))
        max_allocated.assert_called_once_with(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
