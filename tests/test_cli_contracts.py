"""Regression coverage for JSON boundaries, replay, and query ordering."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data, InMemoryDataset
from typer.testing import CliRunner

from gplab.benchmark.identity import compute_comparison_group_key
from gplab.cli import query, replay, run_train_job, train_cli
from gplab.experiment.query import QuerySpec, build_benchmark_report
from gplab.experiment.record import summarize_record
from gplab.graph import ConnectivityType
from gplab.jobs import parse_job
from gplab.jobs.execute import execute_job
from gplab.jobs.job import ExperimentJob


class CliContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = {"experiment": {"dataset": "MUTAG", "pool": {"name": "nopool", "ratio": 0.5},
                           "training": {"runs": 1, "epochs": 1, "patience": 0}}}
        dataset = InMemoryDataset()
        dataset.data, dataset.slices = dataset.collate([
            Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]),
                 y=torch.tensor([i % 2])) for i in range(12)
        ])
        dataset.connectivity_type = ConnectivityType.BINARY
        with patch("gplab.experiment.execute.load_dataset", return_value=dataset), patch(
            "torch.cuda.is_available", return_value=False
        ):
            cls.record = execute_job(parse_job(cls.job), emit_text=False)["record"]

    def setUp(self):
        self.runner = CliRunner()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log = Path(directory.name) / "records.jsonl"
        self.log.write_text(json.dumps(self.record) + "\n", encoding="utf-8")

    def test_job_error_phase_and_field(self):
        invalid = copy.deepcopy(self.job)
        invalid["experiment"]["pool"]["ratio"] = True
        result = self.runner.invoke(run_train_job.app, ["--job-json", json.dumps(invalid)])
        payload = json.loads(result.stdout)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(payload["kind"], "job_error")
        self.assertEqual(payload["error"]["field"], "experiment.pool.ratio")
        with patch("gplab.cli.run_train_job.execute_job", side_effect=RuntimeError("failed")):
            result = self.runner.invoke(run_train_job.app, ["--job-json", json.dumps(self.job)])
        payload = json.loads(result.stdout)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(payload["kind"], "train_error")
        self.assertEqual(payload["error"]["details"]["source"], "job_json")

    def test_job_defaults_are_isolated_and_nested_types_are_strict(self):
        first = parse_job(self.job).to_mapping()
        first["experiment"]["model"]["pre_gnn"].append(999)
        self.assertEqual(parse_job(self.job).experiment.model.pre_gnn, (128,))
        for section, field, value in (("model", "hidden_features", True),
                                      ("model", "pre_gnn", [True]),
                                      ("training", "lr", float("inf"))):
            job = copy.deepcopy(self.job)
            job["experiment"].setdefault(section, {})[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                parse_job(job)
            self.assertTrue(caught.exception.field.startswith(f"experiment.{section}.{field}"))

    def test_job_stdout_is_one_response(self):
        def execute(request, **kwargs):
            print("third-party progress")
            return {"ok": True, "kind": "train_result"}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("gplab.cli.run_train_job.execute_job", side_effect=execute), patch(
            "sys.stdin", io.StringIO(json.dumps(self.job))
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            run_train_job.main(job_file=None, job_json=None, job_stdin=True, output_format="json")
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True, "kind": "train_result"})
        self.assertEqual(stderr.getvalue(), "third-party progress\n")

    def test_replay_executes_once_only_when_requested_in_both_formats(self):
        payload = {"summary": summarize_record(self.record)}
        for output_format in ("json", "text"):
            for run in (False, True):
                with self.subTest(format=output_format, run=run), patch(
                    "gplab.cli.replay.execute_job", return_value=payload
                ) as execute:
                    args = ["--log-file", str(self.log), "--record-id", self.record["record_id"],
                            "--output-format", output_format]
                    result = self.runner.invoke(replay.app, args + (["--run"] if run else []))
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertEqual(execute.call_count, int(run))
                    if run:
                        request = execute.call_args.args[0]
                        self.assertEqual([run.to_mapping() for run in request.fixed_runs],
                                         [{"seed": run["seed"], "split": run["split"]}
                                          for run in self.record["result"]["runs"]])
                        self.assertEqual(execute.call_args.kwargs["emit_text"], output_format == "text")
                    if output_format == "json":
                        response = json.loads(result.stdout)
                        self.assertEqual("rerun" in response, run)
                        self.assertEqual(response["job"]["experiment"]["training"]["seeds"]["mode"], "auto")

    def test_report_sorting_is_stable_and_does_not_mutate_records(self):
        records = [copy.deepcopy(self.record) for _ in range(3)]
        for record, mean in zip(records, (0.25, 0.75, 0.75)):
            record["record_id"] = str(mean)
            record["result"]["mean"] = mean
        records[2]["record_id"] = "tie"
        original = copy.deepcopy(records)
        report = build_benchmark_report(records, QuerySpec(str(self.log)))
        ranked = report["groups"][0]["summaries"]
        self.assertEqual([row["mean"] for row in ranked], [0.75, 0.75, 0.25])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        self.assertEqual(ranked[1]["record_id"], "tie")
        self.assertEqual(records, original)
        result = self.runner.invoke(query.app, ["--log-file", str(self.log), "--report", "--output-format", "json"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.stdout)["kind"], "query_report")


    def test_exported_replay_job_preserves_configuration_splits_and_provenance(self):
        replay_job = ExperimentJob.from_record(self.record, log_file=str(self.log))
        exported = json.loads(json.dumps(replay_job.to_mapping()))
        restored = parse_job(exported)
        self.assertEqual(restored, replay_job)
        self.assertEqual(restored.experiment.to_mapping(), self.record["experiment"])
        self.assertEqual(restored.source_record_id, self.record["record_id"])
        self.assertIsNone(ExperimentJob.from_record(self.record).log_file)
        with patch("gplab.jobs.execute.run_experiment", return_value={
            "environment": self.record["environment"], "result": self.record["result"],
        }):
            payload = execute_job(restored, emit_text=False)
        persisted = json.loads(self.log.read_text().splitlines()[-1])
        self.assertEqual(payload["record"], persisted)
        self.assertEqual(persisted["source_record_id"], self.record["record_id"])
        self.assertNotIn("log_file", persisted)

    def test_benchmark_grouping_uses_actual_splits_and_seeds(self):
        record = copy.deepcopy(self.record)
        expected = compute_comparison_group_key(record)
        record["experiment"]["pool"]["name"] = "topkpool"
        record["experiment"]["training"]["seeds"] = {"mode": "list"}
        record["experiment"]["training"]["activation_checkpoint"] = True
        self.assertEqual(compute_comparison_group_key(record), expected)
        split = record["result"]["runs"][0]["split"]
        split["train"][0], split["test"][0] = split["test"][0], split["train"][0]
        self.assertNotEqual(compute_comparison_group_key(record), expected)

    def test_replay_json_rejects_invalid_index_types_and_unknown_fields(self):
        exported = ExperimentJob.from_record(self.record).to_mapping()
        for change, field in (("boolean_index", "runs[0].split.train[]"),
                              ("unknown_field", "runs[0]")):
            job = copy.deepcopy(exported)
            if change == "boolean_index":
                job["runs"][0]["split"]["train"][0] = True
            else:
                job["runs"][0]["unused"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError) as caught:
                parse_job(job)
            self.assertEqual(caught.exception.field, field)


    def test_record_schema_is_minimal_and_group_key_is_derived(self):
        self.assertEqual(set(self.record), {
            "record_id", "experiment", "environment", "result", "tag", "source_record_id",
        })
        self.assertEqual(set(self.record["environment"]), {
            "python_version", "torch_version", "torch_geometric_version", "tgp_version", "device",
        })
        self.assertFalse(self.record["experiment"]["training"]["activation_checkpoint"])
        summary = summarize_record(self.record)
        self.assertIn("comparison_group_key", summary)
        report = build_benchmark_report([self.record], QuerySpec(str(self.log)))
        self.assertEqual(report["groups"][0]["comparison_group_key"], summary["comparison_group_key"])

    def test_training_checkpoint_round_trips_and_rejects_non_boolean_json(self):
        job = copy.deepcopy(self.job)
        job["experiment"]["training"]["activation_checkpoint"] = True
        parsed = parse_job(job)
        self.assertTrue(parsed.experiment.training.activation_checkpoint)
        self.assertEqual(parse_job(parsed.to_mapping()), parsed)
        record = copy.deepcopy(self.record)
        record["experiment"] = parsed.experiment.to_mapping()
        self.assertTrue(ExperimentJob.from_record(record).experiment.training.activation_checkpoint)
        job["experiment"]["training"]["activation_checkpoint"] = "false"
        with self.assertRaises(ValueError) as caught:
            parse_job(job)
        self.assertEqual(caught.exception.field, "experiment.training.activation_checkpoint")
        with self.assertRaises(ValueError):
            parse_job({**self.job, "execution": {"activation_checkpoint": True}})

    def test_checkpoint_cli_overrides_training_toml(self):
        config = self.log.parent / "experiment.toml"
        config.write_text(
            '[training]\nruns=1\nlr=0.001\nbatch_size=8\npatience=0\nepochs=1\n'
            'activation_checkpoint=true\n[training.split]\ntrain=0.8\nval=0.1\n',
            encoding="utf-8",
        )
        for flag, expected in (([], True), (["--no-activation-checkpoint"], False)):
            with self.subTest(flag=flag), patch(
                "gplab.cli.train_cli.execute_job", return_value={"ok": True},
            ) as execute:
                result = self.runner.invoke(train_cli.app, [
                    "--dataset", "MUTAG", "--experiment-config", str(config),
                    "--output-format", "json", *flag,
                ])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(execute.call_args.args[0].experiment.training.activation_checkpoint, expected)

    def test_replay_environment_check_includes_tgp(self):
        recorded = self.record["environment"]
        current = {**recorded, "tgp_version": "different"}
        status, details = replay._compatibility_status(recorded, current)
        self.assertEqual(status, "mismatch")
        self.assertEqual([item["field"] for item in details if not item["match"]], ["tgp"])
