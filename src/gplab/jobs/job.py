"""Input to CLI execution, including optional replay data and output destination."""
from dataclasses import dataclass

from gplab.benchmark.config import ExperimentConfig
from gplab.benchmark.runs import RunSpec


@dataclass(frozen=True)
class ExperimentJob:
    """An experiment to execute and the boundary-only choices for saving its result."""
    experiment: ExperimentConfig
    log_file: str | None = None
    tag: str | None = None
    fixed_runs: tuple[RunSpec, ...] | None = None
    source_record_id: str | None = None

    @classmethod
    def from_mapping(cls, value: dict) -> "ExperimentJob":
        """Construct a job from JSON already validated by parse_job."""
        runs = value.get("runs")
        return cls(
            experiment=ExperimentConfig.from_mapping(value["experiment"]),
            log_file=value.get("log_file"), tag=value.get("tag"),
            fixed_runs=None if runs is None else tuple(RunSpec.from_mapping(run) for run in runs),
            source_record_id=value.get("source_record_id"),
        )

    @classmethod
    def from_record(cls, record: dict, *, log_file: str | None = None) -> "ExperimentJob":
        """Replay actual seeds and splits, retaining the original requested configuration.

        A replay writes only to the explicitly supplied destination. Its provenance
        is the source record ID; the requested experiment stays unchanged.
        """
        return cls(
            experiment=ExperimentConfig.from_mapping(record["experiment"]),
            log_file=log_file, tag=record["tag"],
            fixed_runs=tuple(RunSpec.from_mapping(run) for run in record["result"]["runs"]),
            source_record_id=record["record_id"],
        )

    def to_mapping(self) -> dict:
        return {
            "experiment": self.experiment.to_mapping(),
            "log_file": self.log_file, "tag": self.tag,
            "runs": None if self.fixed_runs is None else [run.to_mapping() for run in self.fixed_runs],
            "source_record_id": self.source_record_id,
        }
