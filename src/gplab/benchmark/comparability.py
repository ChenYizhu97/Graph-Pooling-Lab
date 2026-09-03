"""Minimal U/W compatibility rules for pooling benchmark construction."""
from __future__ import annotations

from dataclasses import dataclass

from gplab.graph import ConnectivityType
from gplab.layers.conv.profiles import CONV_PROFILES
from gplab.layers.pool.profiles import POOLING_PROFILES, load_pooling_profile


@dataclass(frozen=True)
class ComparabilityResult:
    input_type: ConnectivityType
    output_type: ConnectivityType


def resolve_dataset_connectivity_type(dataset) -> ConnectivityType:
    """Resolve U/W from explicit semantic metadata, never tensor presence alone."""
    declared_type = getattr(dataset, "connectivity_type", None)
    try:
        connectivity_type = ConnectivityType(declared_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Dataset must declare connectivity_type as 'binary' or 'scalar'; "
            "edge_weight tensor presence does not establish graph semantics."
        ) from exc

    if connectivity_type is ConnectivityType.SCALAR:
        for graph in dataset:
            edge_weight = getattr(graph, "edge_weight", None)
            if (
                edge_weight is None
                or edge_weight.dim() != 1
                or edge_weight.numel() != graph.edge_index.size(1)
            ):
                raise ValueError(
                    "A scalar-connectivity dataset must expose one semantic edge_weight value per edge."
                )
    return connectivity_type


def _check_comparability(
    *,
    dataset_type: ConnectivityType,
    pool_name: str,
    pre_conv: str,
    post_conv: str,
) -> tuple[ComparabilityResult | None, str | None]:
    pre_conv_profile = CONV_PROFILES[pre_conv]
    # Scalar values may bypass the pre-conv, but every pre-conv must be able to
    # process binary topology before the unchanged values reach pooling.
    if not pre_conv_profile.can_consume(ConnectivityType.BINARY):
        return None, (
            f"Pre-pooling encoder '{pre_conv}' cannot consume "
            "binary graph topology."
        )

    profile = load_pooling_profile(pool_name)
    output_type = profile.output_type_for(dataset_type)
    if output_type is None:
        return None, (
            f"Pooling profile '{pool_name}' is not declared valid for "
            f"{dataset_type.value}-valued input connectivity."
        )
    if not CONV_PROFILES[post_conv].can_consume(output_type):
        return None, (
            f"{pool_name} produces {output_type.value}-valued pooled connectivity, "
            f"but post-pooling encoder '{post_conv}' cannot consume scalar edge values."
        )
    return ComparabilityResult(dataset_type, output_type), None


def validate_comparability(
    *,
    dataset_type: ConnectivityType,
    pool_name: str,
    pre_conv: str,
    post_conv: str,
) -> ComparabilityResult:
    result, error = _check_comparability(
        dataset_type=dataset_type,
        pool_name=pool_name,
        pre_conv=pre_conv,
        post_conv=post_conv,
    )
    if error is not None:
        raise ValueError(error)
    assert result is not None
    return result


def comparable_pools(
    *,
    dataset_type: ConnectivityType,
    pre_conv: str,
    post_conv: str,
) -> tuple[str, ...]:
    comparable = []
    for pool_name in POOLING_PROFILES:
        result, _ = _check_comparability(
            dataset_type=dataset_type,
            pool_name=pool_name,
            pre_conv=pre_conv,
            post_conv=post_conv,
        )
        if result is not None:
            comparable.append(pool_name)
    return tuple(comparable)
