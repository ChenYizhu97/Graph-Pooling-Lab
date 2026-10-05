"""Example pool registration using keyword parameters and native TGP output."""
from gplab.layers.pool import POOLING_PROFILES, PoolingProfile


def _build_custom_pool(in_channels, avg_node_num=None, **params):
    """Reuse GPLab's configured TopK while accepting its native constructor options."""
    return POOLING_PROFILES["topkpool"].build(in_channels=in_channels, **params)


CUSTOM_POOL_PROFILE = PoolingProfile(
    builder=_build_custom_pool,
    signatures=POOLING_PROFILES["topkpool"].signatures,
)
