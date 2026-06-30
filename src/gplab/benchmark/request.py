from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .case import BenchmarkCase
from .comparison import compute_case_id
from .execution import ExecutionOptions


@dataclass(frozen=True)
class BenchmarkRequest:
    case: BenchmarkCase
    execution: ExecutionOptions

    @classmethod
    def from_mapping(cls, value: dict) -> BenchmarkRequest:
        return cls(
            case=BenchmarkCase.from_mapping(value["case"]),
            execution=ExecutionOptions.from_mapping(value["execution"]),
        )

    @classmethod
    def from_record_for_replay(
        cls,
        record: dict,
        *,
        replay_log_file: Optional[str] = None,
    ) -> BenchmarkRequest:
        return cls(
            case=BenchmarkCase.from_record(record),
            execution=ExecutionOptions.from_record(record, log_file=replay_log_file),
        )

    @property
    def case_id(self) -> str:
        return compute_case_id(self.case)

    def to_mapping(self) -> dict:
        return {
            "case": self.case.to_mapping(),
            "execution": self.execution.to_mapping(),
        }
