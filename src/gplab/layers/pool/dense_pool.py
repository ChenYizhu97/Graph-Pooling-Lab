"""Configured TGP dense poolers with GPLab's fixed-slot sparse output semantics."""
import torch
from tgp.poolers import DiffPool
from tgp.poolers import MinCutPooling as TGPMinCutPooling
from tgp.select import Select, SelectOutput
from tgp.src import PoolingOutput
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import dense_to_sparse


class FixedClusterOutput:
    """Keep all K cluster slots and zero-weight edges when flattening dense TGP output.

    TGP's default sparse conversion removes zero/tiny edges and can remove empty
    clusters. In GPLab, dropping zero diagonal entries would also change GCN's
    automatic self-loop insertion, so this is a model-semantic boundary.
    """
    def _finalize_sparse_output(self, x_pool, adj_pool, batch, batch_pooled, so):
        num_graphs, k = adj_pool.shape[:2]
        edges, _ = dense_to_sparse(torch.ones_like(adj_pool))
        pooled_batch = torch.arange(num_graphs, device=x_pool.device).repeat_interleave(k)
        return x_pool.flatten(0, 1), edges, adj_pool.reshape(-1), pooled_batch


class MinCutPooling(FixedClusterOutput, TGPMinCutPooling):
    """Native TGP MinCut, retaining GPLab's fixed output slots and explicit zero edges."""


class DensePooling(FixedClusterOutput, DiffPool):
    """TGP MLP assignment/reduction/connectivity without auxiliary regularization."""
    def compute_loss(self, adj, S, num_nodes):
        return None


class GraphAssignment(Select):
    """DiffPool's graph-aware assignment, using the existing one-layer DenseGCN scorer."""
    is_dense = True

    def __init__(self, in_channels: int, k: int):
        super().__init__()
        self.gnn = DenseGCNConv(in_channels, k)

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, adj, mask=None):
        assignment = self.gnn(x, adj, mask).softmax(dim=-1)
        if mask is not None:
            assignment = assignment * mask.unsqueeze(-1)
        return SelectOutput(s=assignment, in_mask=mask)


class GraphDiffPool(FixedClusterOutput, DiffPool):
    """TGP DiffPool components with graph-aware assignment and the existing padded loss mean.

    TGP 1.0.2's forward computes MLP assignment internally and ignores supplied
    `so`. This small orchestration passes adjacency to our assignment instead;
    preprocessing, reduction, connection, and auxiliary losses remain TGP's.
    """
    def __init__(self, in_channels: int, k: int):
        super().__init__(in_channels, k, link_loss_coeff=0.1, ent_loss_coeff=0.1,
                         normalize_loss=True, remove_self_loops=False, degree_norm=False,
                         adj_transpose=False, sparse_output=True)
        self.selector = GraphAssignment(in_channels, k)

    def forward(self, x, adj=None, edge_weight=None, so=None, mask=None, batch=None,
                batch_pooled=None, lifting=False, **kwargs):
        if lifting:
            return self.lift(x_pool=x, so=so, batch=batch, batch_pooled=batch_pooled)
        x, adj, mask = self._ensure_batched_inputs(x, adj, edge_weight, batch, mask)
        so = self.select(x=x, adj=adj, mask=mask)
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)
        adj_pooled, _ = self.connect(edge_index=adj, so=so, batch=batch, batch_pooled=batch_pooled)
        # PyG's original entropy mean includes padded slots, unlike TGP's valid-node mean.
        loss = self.compute_loss(adj, so.s, num_nodes=mask.numel())
        x_pooled, edges, weights, batch_pooled = self._finalize_sparse_output(
            x_pooled, adj_pooled, batch, batch_pooled, so,
        )
        return PoolingOutput(x=x_pooled, edge_index=edges, edge_weight=weights,
                             batch=batch_pooled, so=so, loss=loss)
