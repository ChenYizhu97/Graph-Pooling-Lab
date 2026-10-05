"""Optional dataset-level size matching, independent of graph comparability."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite, nextafter


@dataclass(frozen=True)
class CompressionControl:
    """Keep native method parameters or request an approximate retained-node budget."""
    mode: str = "native"
    target_retention: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"native", "matched"}:
            raise ValueError("experiment.compression.mode must be 'native' or 'matched'.")
        target = self.target_retention
        if self.mode == "native":
            if target is not None:
                raise ValueError("experiment.compression.target_retention is only valid in matched mode.")
        elif (isinstance(target, bool) or not isinstance(target, (int, float))
              or not isfinite(target) or not 0 < target <= 1):
            raise ValueError("experiment.compression.target_retention must be in (0, 1] for matched mode.")

    def to_mapping(self) -> dict:
        return asdict(self)


def resolve_compression(name: str, params: dict, control: CompressionControl,
                        avg_input_nodes: float) -> dict:
    """Resolve construction parameters without changing the requested configuration.

    Matching is approximate across a dataset, never a per-graph size guarantee.
    Unknown/custom, identity, and adaptive methods retain their native parameters.
    A matched status means a budget was applied, not that execution achieved it.
    """
    resolved = deepcopy(params)
    status = "native"
    if control.mode == "matched":
        target = control.target_retention
        status = "unmatched"
        if name in {"mincutpool", "diffpool", "densepool"}:
            resolved.pop("ratio", None)
            # A pooling layer needs at least one output cluster, even if rounding gives zero.
            resolved["k"] = max(1, round(target * avg_input_nodes))
            status = "matched"
        elif name in {"topkpool", "sagpool", "asapool", "sparsepool"}:
            adaptive = name in {"topkpool", "sagpool"} and resolved.get("min_score") is not None
            if not adaptive:
                # TGP/PyG interpret 1.0 as a count. The nearest lower float keeps
                # ceil(ratio * N) == N for practical graph sizes at a 100% target.
                resolved["ratio"] = target if target < 1 else nextafter(1.0, 0.0)
                status = "matched"
    return {"status": status, "pool_params": resolved, "avg_input_nodes": avg_input_nodes}
