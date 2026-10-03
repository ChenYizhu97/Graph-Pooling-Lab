"""Register a configured pool without implementing a new pooling algorithm.

Usage:
    gplab-train --pool examples.custom_pool_plugin:CUSTOM_POOL_PROFILE --pool-ratio 0.6
"""
from gplab.layers.pool import POOLING_PROFILES, PoolingProfile


def _build_custom_pool(in_channels, ratio, avg_node_num, _nonlinearity):
    """Reuse GPLab's configured native TGP TopK with tanh fixed for this profile."""
    return POOLING_PROFILES["topkpool"].build(
        in_channels=in_channels, ratio=ratio, avg_node_num=avg_node_num,
        nonlinearity="tanh",
    )


CUSTOM_POOL_PROFILE = PoolingProfile(
    builder=_build_custom_pool,
    signatures=POOLING_PROFILES["topkpool"].signatures,
)
