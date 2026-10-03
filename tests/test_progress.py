"""Verify GPLab's progress content, early-stop semantics, and quiet/log output modes."""
import io
import unittest
from unittest.mock import patch

from rich.console import Console

from gplab.experiment.progress import TrainingProgress


class TrainingProgressTests(unittest.TestCase):
    def test_redirected_output_contains_only_run_start_and_final_summary(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=160)
        with TrainingProgress("MUTAG · SAGPool · cpu", 2, 50, enabled=True, console=console) as progress:
            progress.start_run(1, 7)
            progress.stage("validation")
            progress.update_epoch(12, val_loss=0.6, best_loss=0.5, best_epoch=9,
                                  stale_epochs=3, patience=2)
            self.assertNotIn("test acc", output.getvalue())
            self.assertNotIn("Val loss", output.getvalue())
            progress.finish_run({"best_epoch": 9, "test_acc": 0.75, "epochs_trained": 12,
                                 "training_wall_time_seconds": 1.25})
        text = output.getvalue()
        self.assertNotIn("\x1b", text)
        self.assertIn("seed=7", text)
        self.assertIn("test acc=75.00%", text)
        self.assertIn("early stopped", text)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertEqual(progress.tasks[progress.runs_task].completed, 1)
        self.assertEqual(progress.tasks[progress.epochs_task].completed, 12)
        self.assertEqual(progress.tasks[progress.epochs_task].total, 50)

    def test_terminal_view_shows_budget_stage_and_actual_patience_rule(self):
        output = io.StringIO()
        progress = TrainingProgress("MUTAG · SAGPool · cpu", 3, 100, enabled=True,
                                    console=Console(file=output, force_terminal=True, width=100))
        progress.start_run(2, 7)
        progress.stage("validation")
        progress.update_epoch(4, val_loss=0.6, best_loss=0.5, best_epoch=2,
                              stale_epochs=2, patience=2)
        console = Console(file=output, force_terminal=False, width=100)
        console.print(progress)
        text = output.getvalue()
        for label in ("Runs completed", "Epochs (max)", "validation", "Run 2/3", "seed=7",
                      "Val loss 0.6000", "Best 0.5000", "stop > 2"):
            self.assertIn(label, text)
        self.assertNotIn("test acc", text)

    def test_disabled_progress_is_silent_and_defaults_to_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            with TrainingProgress("MUTAG", 1, 1, enabled=False) as progress:
                self.assertIs(progress.console.file, stderr)
                progress.start_run(1, 7)
                progress.stage("training")
                progress.finish_run({"epochs_trained": 1, "best_epoch": 1,
                                     "test_acc": 0.5, "training_wall_time_seconds": 1})
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
