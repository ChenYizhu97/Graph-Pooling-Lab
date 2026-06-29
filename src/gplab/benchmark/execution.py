from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecutionOptions:
    log_file: Optional[str] = None
    tag: Optional[str] = None
    activation_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, value: dict | None) -> ExecutionOptions:
        payload = value or {}
        return cls(
            log_file=payload.get("log_file"),
            tag=payload.get("tag"),
            activation_checkpoint=bool(payload.get("activation_checkpoint", False)),
        )

    @classmethod
    def from_record(cls, record: dict, *, log_file: Optional[str] = None) -> ExecutionOptions:
        recorded = cls.from_mapping(record.get("execution"))
        return cls(
            log_file=log_file,
            tag=recorded.tag,
            activation_checkpoint=recorded.activation_checkpoint,
        )

    def to_mapping(self) -> dict:
        return {
            "log_file": self.log_file,
            "tag": self.tag,
            "activation_checkpoint": self.activation_checkpoint,
        }

    def request_metadata(self) -> dict:
        return self.to_mapping()
