"""GPLab run resolution and replay validation, independent of model/backend behavior."""
import unittest
from dataclasses import replace
from unittest.mock import patch

from gplab.benchmark.runs import RunSpec, SplitIndices, resolve_runs
from gplab.jobs import parse_job


def training_config():
    return parse_job({"experiment": {
        "dataset": "MUTAG", "pool": {"name": "nopool", "ratio": 0.5},
        "training": {"runs": 1, "epochs": 1, "patience": 0},
    }}).experiment.training


class RunResolutionTests(unittest.TestCase):
    def test_generated_runs_pair_seeds_with_complete_partitions(self):
        training = replace(training_config(), runs=3)
        runs = resolve_runs(training, dataset_size=12)
        self.assertEqual(runs, resolve_runs(training, dataset_size=12))
        self.assertEqual(len(runs), 3)
        self.assertEqual(len({run.seed for run in runs}), 3)
        for run in runs:
            self.assertEqual(RunSpec.from_mapping(run.to_mapping()), run)
            self.assertEqual(sorted((*run.split.train, *run.split.val, *run.split.test)), list(range(12)))

    def test_fixed_runs_bypass_generation_without_rewriting_policy(self):
        training = training_config()
        fixed = (RunSpec(7, SplitIndices((2, 3), (1,), (0,))),)
        with patch("gplab.benchmark.runs.resolve_seeds") as seeds, patch(
            "gplab.benchmark.runs.build_split_indices"
        ) as splits:
            self.assertIs(resolve_runs(training, 4, fixed), fixed)
        seeds.assert_not_called()
        splits.assert_not_called()
        self.assertEqual(training.seeds.mode, "auto")

    def test_wrong_run_count_cannot_silently_skip_training(self):
        run = RunSpec(7, SplitIndices((0, 1), (2,), (3,)))
        for runs in ((), (run, run)):
            with self.subTest(count=len(runs)), self.assertRaisesRegex(ValueError, "Run count"):
                resolve_runs(training_config(), 4, runs)

    def test_invalid_partitions_fail_before_training(self):
        invalid = (
            SplitIndices((), (1,), (0, 2, 3)),
            SplitIndices((0, 0), (1,), (2, 3)),
            SplitIndices((0, 1), (1,), (2, 3)),
            SplitIndices((0,), (1,), (2,)),
            SplitIndices((0, 4), (1,), (2, 3)),
            SplitIndices((False,), (1,), (2, 3)),
        )
        for split in invalid:
            with self.subTest(split=split), self.assertRaises(ValueError):
                resolve_runs(training_config(), 4, (RunSpec(7, split),))
