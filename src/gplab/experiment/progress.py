"""Human-facing run/epoch progress; redirected output uses plain run summaries."""
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text


class TrainingProgress(Progress):
    """Display completed runs, the current epoch budget, and validation metrics.

    Epoch totals are upper limits because early stopping can end a run sooner.
    Test accuracy is displayed only after final evaluation. All output goes to
    stderr, and disabled progress emits nothing (the JSON execution path).
    """
    def __init__(
        self, label: str, runs: int, epochs: int, *, enabled: bool,
        console: Console | None = None,
    ) -> None:
        self.label = label
        self.run_count = runs
        self.epoch_limit = epochs
        self.enabled = enabled
        self.run_status = ""
        self.metrics = ""
        console = console or Console(stderr=True)
        super().__init__(
            TextColumn("{task.description}"), BarColumn(),
            MofNCompleteColumn(), TimeElapsedColumn(),
            console=console, disable=not enabled or not console.is_terminal,
            transient=True, refresh_per_second=4,
            redirect_stdout=False, redirect_stderr=False,
        )
        self.runs_task = self.add_task("Runs completed", total=runs)
        self.epochs_task = self.add_task("Epochs (max)", total=epochs, start=False)

    def get_renderables(self):
        """Keep task counts and metrics in separate rows so narrow terminals can wrap."""
        yield Text(self.label, style="bold")
        yield from super().get_renderables()
        yield Text(self.run_status)
        yield Text(self.metrics)

    def start_run(self, index: int, seed: int) -> None:
        self.run_status = f"Run {index}/{self.run_count} · seed={seed}"
        self.metrics = ""
        self.reset(self.epochs_task, total=self.epoch_limit)
        if self.enabled and not self.console.is_terminal:
            self.console.print(f"{self.label} · {self.run_status}", markup=False)

    def stage(self, name: str) -> None:
        self.update(self.epochs_task, description=f"Epochs (max) · {name}")

    def update_epoch(
        self, epoch: int, *, val_loss: float, best_loss: float,
        best_epoch: int, stale_epochs: int, patience: int,
    ) -> None:
        self.update(self.epochs_task, completed=epoch)
        # GPLab stops after patience + 1 non-improving epochs, not at patience.
        self.metrics = (
            f"Val loss {val_loss:.4f}   Best {best_loss:.4f} @ epoch {best_epoch}   "
            f"No improvement {stale_epochs} (stop > {patience})"
        )

    def finish_run(self, result: dict) -> None:
        self.stop_task(self.epochs_task)
        self.advance(self.runs_task)
        if self.enabled:
            reason = ("early stopped" if result["epochs_trained"] < self.epoch_limit
                      else "epoch limit reached")
            self.console.print(
                f"✓ {self.run_status} · best epoch={result['best_epoch']} · "
                f"test acc={result['test_acc']:.2%} · "
                f"training={result['training_wall_time_seconds']:.1f}s · {reason}",
                markup=False,
            )
