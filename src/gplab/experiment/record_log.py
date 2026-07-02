from __future__ import annotations

from gplab.experiment.identity import require_record_id
from gplab.experiment.record import ExperimentRecord
from gplab.utils.jsonl import read_jsonl


RECORD_FIELDS = ("case", "execution", "run_plan", "runtime", "result")


class RecordLogError(ValueError):
    def __init__(self, message: str, *, field: str | None = None, expected: str | None = None) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected


def require_experiment_record(record: dict) -> ExperimentRecord:
    try:
        ensured = require_record_id(record)
    except ValueError as exc:
        raise RecordLogError(str(exc), field="record_id", expected="record_id") from exc

    missing = [field for field in RECORD_FIELDS if field not in ensured]
    if missing:
        raise RecordLogError(
            f"Record is missing required field(s): {', '.join(missing)}.",
            field="record",
            expected=f"fields: {', '.join((*RECORD_FIELDS, 'record_id'))}",
        )
    return ensured


def load_record_log(log_file: str) -> list[ExperimentRecord]:
    return [require_experiment_record(record) for record in read_jsonl(log_file)]


def find_record_by_id(records: list[ExperimentRecord], record_id: str) -> ExperimentRecord:
    record = next((record for record in records if record["record_id"] == record_id), None)
    if record is None:
        raise RecordLogError(
            f"record_id '{record_id}' was not found in the selected log file.",
            field="record_id",
            expected="existing record_id",
        )
    return record
