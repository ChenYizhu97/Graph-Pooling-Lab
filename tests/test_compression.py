"""Compression protocols, actual retention, and their separation from comparability."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data, InMemoryDataset
from typer.testing import CliRunner

from gplab.benchmark import ComparisonSetting, CompressionControl, check_comparability
from gplab.benchmark.compression import resolve_compression
from gplab.benchmark.config import PoolConfig
from gplab.benchmark.identity import compute_comparison_group_key
from gplab.cli import train_cli
from gplab.data.profiles import DatasetProfile
from gplab.experiment.query import QuerySpec, build_benchmark_report, format_report_text
from gplab.graph import ConnectivityType
from gplab.jobs import parse_job
from gplab.jobs.execute import execute_job
from gplab.jobs.job import ExperimentJob


class CompressionTests(unittest.TestCase):
    def test_native_preserves_all_method_parameters(self):
        for name, params in (("topkpool", {"ratio": 0.3}), ("diffpool", {"k": 7}),
                             ("topkpool", {"min_score": 0.4}), ("custom:pool", {"budget": [2, 3]})):
            with self.subTest(name=name):
                original = copy.deepcopy(params)
                resolved = resolve_compression(name, params, CompressionControl(), 11)
                self.assertEqual(resolved["status"], "native")
                self.assertEqual(resolved["pool_params"], original)
                self.assertEqual(params, original)
                self.assertIsNot(resolved["pool_params"], params)

    def test_matching_resolves_ratios_and_rounds_fixed_k(self):
        control = CompressionControl("matched", 0.5)
        for name in ("topkpool", "sagpool", "asapool", "sparsepool"):
            resolved = resolve_compression(name, {"ratio": 0.2, "nonlinearity": "relu"}, control, 5.5)
            self.assertEqual(resolved["pool_params"], {"ratio": 0.5, "nonlinearity": "relu"})
            self.assertEqual(resolved["status"], "matched")
        for name in ("diffpool", "mincutpool", "densepool"):
            for params in ({"ratio": 0.2}, {"k": 99}):
                resolved = resolve_compression(name, params, control, 5.5)
                self.assertEqual(resolved["pool_params"], {"k": 3})
                self.assertEqual(resolved["status"], "matched")
        self.assertEqual(resolve_compression("diffpool", {}, control, 1)["pool_params"], {"k": 1})

    def test_adaptive_identity_and_unknown_methods_can_remain_unmatched(self):
        for name, params in (("topkpool", {"ratio": 0.2, "min_score": 0.4}),
                             ("sagpool", {"min_score": 0.4}), ("nopool", {}),
                             ("custom:pool", {"threshold": 0.1})):
            resolved = resolve_compression(name, params, CompressionControl("matched", 0.5), 10)
            self.assertEqual(resolved["status"], "unmatched")
            self.assertEqual(resolved["pool_params"], params)

    def test_target_one_does_not_become_one_node(self):
        resolved = resolve_compression("topkpool", {}, CompressionControl("matched", 1.0), 10)
        ratio = resolved["pool_params"]["ratio"]
        self.assertLess(ratio, 1)
        self.assertEqual(torch.ceil(torch.tensor([3., 8.], dtype=torch.float64) * ratio).tolist(), [3., 8.])

    def test_configuration_and_cli_validation(self):
        base = {"experiment": {"dataset": "MUTAG", "pool": {"name": "topkpool"},
                               "training": {"num_runs": 1, "epochs": 1, "patience": 0}}}
        for compression in ({"mode": "other"}, {"mode": "matched"},
                            {"mode": "native", "target_retention": .5},
                            {"mode": "matched", "target_retention": True},
                            {"mode": "matched", "target_retention": 0},
                            {"mode": "matched", "target_retention": 1.1},
                            {"mode": "matched", "target_retention": .5, "extra": 1}):
            job = copy.deepcopy(base)
            job["experiment"]["compression"] = compression
            with self.subTest(compression=compression), self.assertRaises(ValueError):
                parse_job(job)
        with patch("gplab.cli.train_cli.execute_job", return_value={"ok": True}) as execute:
            result = CliRunner().invoke(train_cli.app, [
                "--pool", "diffpool", "--compression-mode", "matched", "--target-retention", "0.4",
                "--output-format", "json",
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].experiment.compression, CompressionControl("matched", 0.4))

    def test_execution_records_budget_and_reports_achieved_retention(self):
        source = InMemoryDataset()
        source.data, source.slices = source.collate([
            Data(x=torch.ones(n, 2), edge_index=torch.stack((torch.arange(n), torch.arange(n).roll(1))),
                 y=torch.tensor([i % 2])) for i, n in enumerate([3, 8] * 6)
        ])
        profile = DatasetProfile(lambda: source, ConnectivityType.BINARY)
        with patch("gplab.data.profiles.DATASET_PROFILES", {"fixture": profile}), patch(
            "torch.cuda.is_available", return_value=False
        ):
            records = []
            for name, params, status in (("topkpool", {"ratio": .1}, "matched"),
                                        ("densepool", {"k": 99}, "matched"),
                                        ("topkpool", {"min_score": .2}, "unmatched")):
                job = parse_job({"experiment": {
                    "dataset": "fixture", "pool": {"name": name, "params": params},
                    "compression": {"mode": "matched", "target_retention": .5},
                    "model": {"hidden_features": 4, "pre_gnn": [4], "post_gnn": [8, 4]},
                    "training": {"num_runs": 1, "epochs": 1, "patience": 0},
                }})
                record = execute_job(job, emit_text=False)["record"]
                records.append(record)
                self.assertEqual(record["experiment"]["pool"]["params"], params)
                self.assertEqual(record["result"]["compression"]["status"], status)
                self.assertEqual(record["result"]["compression"]["avg_input_nodes"], 5.5)
                replay = parse_job(json.loads(json.dumps(ExperimentJob.from_record(record).to_mapping())))
                self.assertEqual(replay.experiment, job.experiment)
                repeated = execute_job(replay, emit_text=False)["record"]
                self.assertEqual(repeated["result"]["compression"], record["result"]["compression"])
                self.assertEqual(repeated["result"]["runs"][0]["structural_stats"],
                                 record["result"]["runs"][0]["structural_stats"])
            setting = ComparisonSetting("fixture", job.experiment.model)
            self.assertTrue(check_comparability([PoolConfig("topkpool", {"min_score": .2}),
                                                PoolConfig("densepool", {"k": 99})], setting).comparable)
            report = build_benchmark_report(records, QuerySpec("unused"))
            self.assertEqual(len(report["groups"]), 1)
            for summary in report["groups"][0]["summaries"]:
                record = next(item for item in records if item["record_id"] == summary["record_id"])
                run = record["result"]["runs"][0]
                self.assertEqual(summary["mean_node_retention"], run["structural_stats"]["mean_node_retention"])
                self.assertEqual(summary["mean_training_wall_time_seconds"], run["training_wall_time_seconds"])
                self.assertIsNone(summary["peak_training_cuda_allocated_bytes"])
            dense_stats = records[1]["result"]["runs"][0]["structural_stats"]
            self.assertEqual(dense_stats["total_output_nodes"], 3 * dense_stats["num_graphs"])
            # Different source sizes share one K; measured retention need not equal the target.
            self.assertNotEqual(dense_stats["mean_node_retention"], .5)
            self.assertIn("compression_status=unmatched", format_report_text(report))
            changed = copy.deepcopy(records[0])
            changed["experiment"]["pool"]["params"] = {"ratio": .9}
            self.assertEqual(compute_comparison_group_key(changed), compute_comparison_group_key(records[0]))
            changed["experiment"]["compression"]["target_retention"] = .25
            self.assertNotEqual(compute_comparison_group_key(changed), compute_comparison_group_key(records[0]))
