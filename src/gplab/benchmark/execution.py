"""Execution choices that do not change the requested experiment configuration."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionOptions:
    """Control model execution; activation checkpointing trades compute for memory."""
    activation_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, value: dict) -> "ExecutionOptions":
        return cls(**value)

    def to_mapping(self) -> dict:
        return asdict(self)
