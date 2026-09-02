from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .case import BenchmarkCase
from .identity import compute_case_id
from .execution import ExecutionOptions
from .plan import RunPlan


@dataclass(frozen=True)
class BenchmarkRequest:
    case: BenchmarkCase
    execution: ExecutionOptions
    fixed_run_plan: Optional[RunPlan] = None

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
            fixed_run_plan=RunPlan.from_mapping(record["run_plan"]),
        )

    @property
    def case_id(self) -> str:
        return compute_case_id(self.case)

    def to_mapping(self) -> dict:
        return {
            "case": self.case.to_mapping(),
            "execution": self.execution.to_mapping(),
        }
