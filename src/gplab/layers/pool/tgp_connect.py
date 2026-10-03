"""Compatibility fix for TGP 1.0.2's node-selection edge ordering."""
from tgp.connect import SparseConnect


class SelectionConnect(SparseConnect):
    """Keep induced edges indexed by the same clusters as TGP's reduced features."""

    def forward(self, edge_index, so, **kwargs):
        edges, weights = super().forward(edge_index, so, **kwargs)
        # Partial selection labels edges in source-node order in TGP 1.0.2,
        # but the reducer emits cluster order. Full selection already remaps.
        if so.num_supernodes < so.num_nodes:
            edges = so.cluster_index[edges]
        return edges, weights
