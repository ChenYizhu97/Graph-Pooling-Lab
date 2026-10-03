"""Check one pool's declared graph domains against a dataset and encoder path."""
from gplab.graph import ConnectivityType
from gplab.layers.conv.profiles import CONV_PROFILES
from gplab.layers.pool.profiles import POOLING_PROFILES, load_pooling_profile


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


def pool_compatibility_error(
    *,
    dataset_type: ConnectivityType,
    pool_name: str,
    pre_conv: str,
    post_conv: str,
) -> str | None:
    """Return a domain incompatibility reason, or None for a compatible path.

    The dataset type must belong to the pool's input domains. Every possible
    output for that input must belong to the post-convolution's capabilities;
    one supported alternative is not enough to make the path safe.
    Unknown profiles/encoders are configuration errors and raise instead.
    """
    if not isinstance(dataset_type, ConnectivityType):
        raise TypeError("dataset_type must be a ConnectivityType.")
    # Scalar values may bypass a topology-only pre-conv and reach pooling
    # unchanged; pre-conv support therefore must not narrow the pool signature.
    if not CONV_PROFILES[pre_conv].can_consume(ConnectivityType.BINARY):
        return f"Pre-pooling encoder '{pre_conv}' cannot consume binary graph topology."

    profile = load_pooling_profile(pool_name)
    if dataset_type not in profile.input_types:
        return (
            f"Pooling profile '{pool_name}' is not declared valid for "
            f"{dataset_type.value}-valued input connectivity."
        )
    unsupported = profile.output_types_for(dataset_type) - CONV_PROFILES[post_conv].connectivity_types
    if unsupported:
        # Stable diagnostics even when signatures were declared as an unordered set.
        types = ", ".join(sorted(value.value for value in unsupported))
        return (
            f"{pool_name} produces {types}-valued pooled connectivity, "
            f"but post-pooling encoder '{post_conv}' cannot consume {types} edge values."
        )
    return None


def validate_pool_compatibility(
    *,
    dataset_type: ConnectivityType,
    pool_name: str,
    pre_conv: str,
    post_conv: str,
) -> None:
    """Reject an incompatible single-pool path before model construction."""
    error = pool_compatibility_error(
        dataset_type=dataset_type, pool_name=pool_name, pre_conv=pre_conv, post_conv=post_conv,
    )
    if error is not None:
        raise ValueError(error)


def compatible_pools(
    *,
    dataset_type: ConnectivityType,
    pre_conv: str,
    post_conv: str,
) -> tuple[str, ...]:
    """List compatible built-ins; this is discovery, not a comparison verdict."""
    return tuple(
        name for name in POOLING_PROFILES
        if pool_compatibility_error(
            dataset_type=dataset_type, pool_name=name, pre_conv=pre_conv, post_conv=post_conv,
        ) is None
    )
